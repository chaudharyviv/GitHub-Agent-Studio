"""
The tools an agent can call: GitHub tools (bound to one repository) plus memory tools.

A ``Toolbox`` does three jobs for the agent loop:
  1. ``specs()``: the OpenAI function-calling definitions
  2. ``call()``: parse + validate the model's JSON arguments, run the tool, and
     return an outcome (never raises; failures become ToolErrors the model can read)
  3. record side effects in memory (e.g. saving the repository profile)

The repository is fixed per toolbox, so the model cannot wander to another repo:
``owner`` and ``repo`` are hidden from the schemas and injected on every call.
"""

import json
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

from pydantic import BaseModel, Field, ValidationError

import tools
from agents.limits import Limits, get_limits
from memory import Finding, MemoryStore, RepositoryProfile, UserContext
from memory.schemas import Severity
from tools import schemas as s

# Size arguments the model may not choose: the toolbox sets them from its Limits (see agents/limits.py).
_LIMIT_ARGUMENTS = {
    "get_file_content": ("max_chars", "file_chars"),
    "get_dependency_files": ("max_chars_per_file", "dependency_file_chars"),
    "get_repository_tree": ("max_entries", "tree_entries"),
}


class SaveFindingArgs(BaseModel):
    severity: Severity = Field(..., description="'info' for facts, 'warning' for risks/smells, 'critical' for serious problems")
    category: str = Field(..., description="Short label, e.g. architecture, dependency_risk, code_quality, health, documentation")
    finding: str = Field(..., description="One or two sentences stating the observation")
    evidence: Optional[str] = Field(None, description="Concrete support: file paths, numbers, dates")
    confidence: float = Field(0.8, ge=0.0, le=1.0)


class RecallFindingsArgs(BaseModel):
    category: Optional[str] = None
    severity: Optional[Severity] = None
    agent: Optional[str] = None
    limit: int = Field(20, ge=1, le=50)


class SaveUserContextArgs(BaseModel):
    key: str = Field(..., description="Short snake_case key, e.g. learning_focus")
    value: str = Field(..., description="What the user asked you to remember")


class MemoryWriteResult(BaseModel):
    saved: bool
    message: str


class RecalledFindings(BaseModel):
    findings: list[dict]


class Notice(BaseModel):
    note: str


# name -> (handler in tools.github, input model, LLM-facing description)
_GITHUB_TOOLS = {
    "get_repository": (tools.get_repository, s.RepositoryInput, "Repository metadata: description, stars, forks, language, topics, license, default branch, last push."),
    "get_repository_tree": (tools.get_repository_tree, s.RepositoryTreeInput, "List files/directories. Defaults to the root; pass `path` to look inside a directory, `recursive` (+ `max_depth`) to go deeper."),
    "get_file_content": (tools.get_file_content, s.FileContentInput, "Read one text file (e.g. README.md, a config or source file). Long files are truncated."),
    "get_dependency_files": (tools.get_dependency_files, s.DependencyFilesInput, "Find and read dependency manifests (package.json, requirements.txt, pyproject.toml, go.mod, Cargo.toml, ...)."),
    "get_issues": (tools.get_issues, s.IssueInput, "List issues (pull requests excluded), filterable by state and labels."),
    "get_pull_requests": (tools.get_pull_requests, s.PullRequestInput, "List pull requests: open, merged, closed-unmerged, or all."),
    "get_commits": (tools.get_commits, s.CommitInput, "Recent commits, newest first; `since` (ISO date) limits how far back."),
    "get_releases": (tools.get_releases, s.ReleaseInput, "Published releases with notes and assets."),
    "get_contributors": (tools.get_contributors, s.ContributorInput, "Top contributors by commit count."),
    "search_code": (tools.search_code, s.SearchCodeInput, "Search code in this repository (needs a GitHub token)."),
}

_MEMORY_TOOLS = {
    "save_finding": (SaveFindingArgs, "Save a notable observation to long-term memory so it is available in future conversations."),
    "recall_findings": (RecallFindingsArgs, "Look up previously saved findings for this repository, optionally filtered."),
    "save_user_context": (SaveUserContextArgs, "Remember something the user told you (e.g. what they are learning). Same key overwrites."),
}


@dataclass
class ToolOutcome:
    name: str
    arguments: dict
    content: str  # JSON text handed back to the model
    data: Any  # same result as a plain dict, for display
    summary: str  # one line for the UI
    is_error: bool = False
    memory_note: Optional[str] = None  # set when the call wrote to memory


def _llm_schema(model: type[BaseModel], hidden: Iterable[str] = ()) -> dict:
    schema = model.model_json_schema()
    for key in hidden:
        schema["properties"].pop(key, None)
    required = [r for r in schema.get("required", []) if r not in hidden]
    schema.pop("required", None)
    if required:
        schema["required"] = required
    schema.pop("title", None)
    schema.pop("description", None)
    return schema


class Toolbox:
    def __init__(
        self,
        owner: str,
        repo: str,
        store: MemoryStore,
        session_id: str,
        agent_name: str = "single_agent",
        include: Optional[Iterable[str]] = None,
        categories: Optional[Iterable[str]] = None,
        limits: Optional[Limits] = None,
    ):
        """
        ``include`` limits which tools are offered; ``categories`` limits which finding
        categories save_finding accepts (the model is told the allowed list when it slips);
        ``limits`` caps result sizes (defaults to LITE_MODE-aware limits).
        """
        self.limits = limits or get_limits()
        self.owner, self.repo = owner, repo
        self.categories = tuple(categories) if categories is not None else None
        self._fetched: set[tuple[str, str]] = set()  # GitHub calls already answered in this toolbox
        self.store, self.session_id, self.agent_name = store, session_id, agent_name
        self.repo_id = MemoryStore.make_repo_id(owner, repo)
        self._handlers: dict[str, Callable[[dict], Any]] = {}
        self._specs: dict[str, dict] = {}

        for name, (fn, model, description) in _GITHUB_TOOLS.items():
            size_arg = _LIMIT_ARGUMENTS.get(name)
            self._register(name, description, _llm_schema(model, hidden=("owner", "repo", *(size_arg[:1] if size_arg else ()))),
                           lambda args, fn=fn, model=model, name=name: fn(model(**self._bound_arguments(name, args))))
        for name, (model, description) in _MEMORY_TOOLS.items():
            self._register(name, description, _llm_schema(model),
                           lambda args, name=name, model=model: getattr(self, f"_{name}")(model(**args)))
        if include is not None:
            wanted = set(include)
            self._specs = {n: sp for n, sp in self._specs.items() if n in wanted}

    def _bound_arguments(self, name: str, args: dict) -> dict:
        """Model-supplied arguments, with the repository fixed and sizes capped by our limits."""
        bound = {**args, "owner": self.owner, "repo": self.repo}  # bound repo always wins
        if "limit" in bound and isinstance(bound["limit"], int):
            bound["limit"] = min(bound["limit"], self.limits.list_limit)
        if name in _LIMIT_ARGUMENTS:
            argument, limit_field = _LIMIT_ARGUMENTS[name]
            bound[argument] = getattr(self.limits, limit_field)
        return bound

    def _register(self, name: str, description: str, schema: dict, handler: Callable[[dict], Any]) -> None:
        self._specs[name] = {"type": "function", "function": {"name": name, "description": description, "parameters": schema}}
        self._handlers[name] = handler

    def specs(self, only: Optional[Iterable[str]] = None) -> list[dict]:
        """OpenAI ``tools=`` parameter; ``only`` narrows it to the named tools."""
        wanted = set(only) if only is not None else None
        return [spec for name, spec in self._specs.items() if wanted is None or name in wanted]

    def call(self, name: str, raw_arguments: str) -> ToolOutcome:
        """Run one tool call exactly as the model requested it. Never raises."""
        arguments: dict = {}
        try:
            if name not in self._specs:
                raise ValueError(f"unknown tool {name!r}; available: {', '.join(self._specs)}")
            arguments = json.loads(raw_arguments or "{}")
            if not isinstance(arguments, dict):
                raise ValueError("arguments must be a JSON object")
            call_key = (name, json.dumps(arguments, sort_keys=True))
            if call_key in self._fetched:  # identical GitHub call: don't spend context on the same data twice
                result = Notice(note="You already fetched exactly this earlier in this investigation; reuse that result instead of calling again.")
            else:
                result = self._handlers[name](arguments)
                if name in _GITHUB_TOOLS and not isinstance(result, s.ToolError):
                    self._fetched.add(call_key)
        except (ValueError, ValidationError) as exc:  # includes JSONDecodeError
            result = s.ToolError(kind="invalid_input", message=f"Bad arguments for {name}: {exc}")
        except Exception as exc:
            result = s.ToolError(kind="unexpected", message=f"{name} failed: {exc.__class__.__name__}: {exc}")

        note = self._after_call(name, result)
        is_error = isinstance(result, s.ToolError)
        content = result.model_dump_json(exclude_none=True)
        cap = self.limits.tool_result_chars
        if len(content) > cap:
            content = content[:cap] + f"... [truncated {len(content) - cap} chars]"
        return ToolOutcome(
            name=name,
            arguments=arguments,
            content=content,
            data=result.model_dump(exclude_none=True),
            summary=f"{result.kind}: {result.message}" if is_error else f"ok ({len(content):,} chars)",
            is_error=is_error,
            memory_note=note,
        )

    # -- memory side effects and memory tools --------------------------------

    def _after_call(self, name: str, result: Any) -> Optional[str]:
        if isinstance(result, MemoryWriteResult):
            return result.message if result.saved else None
        if name == "get_repository" and isinstance(result, s.RepositoryOutput):
            self.store.save_repository_profile(RepositoryProfile(
                owner=result.owner, name=result.name, description=result.description, stars=result.stars,
                primary_language=result.language, topics=result.topics, license=result.license,
                default_branch=result.default_branch, profile_json=result.model_dump_json(),
            ))
            return "Saved repository profile"
        return None

    def _save_finding(self, args: SaveFindingArgs) -> MemoryWriteResult:
        if self.categories is not None and args.category not in self.categories:
            raise ValueError(f"category {args.category!r} is not allowed for you. Call save_finding again with the same finding and one of these categories: {', '.join(self.categories)}")
        for existing in self.store.get_findings(self.repo_id, category=args.category):
            if existing.finding.strip().lower() == args.finding.strip().lower():
                return MemoryWriteResult(saved=False, message=f"Already recorded as finding #{existing.id}; not saved again.")
        finding_id = self.store.save_finding(Finding(
            session_id=self.session_id, agent=self.agent_name, severity=args.severity, category=args.category,
            finding=args.finding, evidence_json=json.dumps({"evidence": args.evidence} if args.evidence else {}),
            confidence=args.confidence,
        ))
        return MemoryWriteResult(saved=True, message=f"Saved finding #{finding_id} [{args.severity}/{args.category}]: {args.finding}")

    def _recall_findings(self, args: RecallFindingsArgs) -> RecalledFindings:
        found = self.store.get_findings(self.repo_id, agent=args.agent, severity=args.severity, category=args.category, limit=args.limit)
        return RecalledFindings(findings=[
            {"id": f.id, "date": f"{f.created_at:%Y-%m-%d}", "agent": f.agent, "severity": f.severity,
             "category": f.category, "finding": f.finding, "evidence": f.evidence_json}
            for f in found
        ])

    def _save_user_context(self, args: SaveUserContextArgs) -> MemoryWriteResult:
        self.store.save_user_context(UserContext(repository_id=self.repo_id, key=args.key, value=args.value))
        return MemoryWriteResult(saved=True, message=f"Remembered {args.key} = {args.value}")

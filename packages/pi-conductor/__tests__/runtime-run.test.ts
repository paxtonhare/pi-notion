import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { ModelRuntime } from "@earendil-works/pi-coding-agent";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const codingAgentMocks = vi.hoisted(() => ({
  createAgentSession: vi.fn(),
  openSession: vi.fn(),
}));

vi.mock("@earendil-works/pi-coding-agent", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@earendil-works/pi-coding-agent")>();
  return {
    ...actual,
    SessionManager: {
      ...actual.SessionManager,
      open: codingAgentMocks.openSession,
    },
    createAgentSession: codingAgentMocks.createAgentSession,
  };
});

import {
  buildRunScopedConductorTools,
  buildTaskContractPrompt,
  extractFinalAssistantText,
  getWorkerRunRuntimeBackend,
  mapStopReasonToRunOutcome,
  preflightWorkerRunRuntime,
  runWorkerPromptRuntime,
} from "../extensions/runtime.js";
import { mapRunStatusToRuntimeStatus } from "../extensions/runtime-metadata.js";

const createModelRuntime = ModelRuntime.create.bind(ModelRuntime);

describe("worker run runtime helpers", () => {
  beforeEach(() => {
    vi.spyOn(ModelRuntime, "create").mockResolvedValue({ getAvailable: async () => [] } as unknown as ModelRuntime);
  });
  afterEach(() => {
    vi.restoreAllMocks();
    codingAgentMocks.openSession.mockReset();
    codingAgentMocks.createAgentSession.mockReset();
  });

  it("builds an explicit task contract prompt for child worker runs", () => {
    const prompt = buildTaskContractPrompt({
      taskId: "task-1",
      runId: "run-1",
      taskRevision: 2,
      goal: "Implement durable tasks",
      constraints: ["Do not publish a PR"],
      explicitCompletionTools: true,
    });

    expect(prompt).toContain("task-1");
    expect(prompt).toContain("run-1");
    expect(prompt).toContain("revision 2");
    expect(prompt).toContain("Implement durable tasks");
    expect(prompt).toContain("Do not publish a PR");
    expect(prompt).toContain("conductor_child_complete");
    expect(prompt).toContain("conductor_child_progress");
    expect(prompt).toContain("conductor_child_create_gate");
    expect(prompt).toContain("idempotencyKey");
    expect(prompt).not.toContain("conductor_child_create_followup_task");
  });

  it("includes follow-up task instructions only when allowed", () => {
    const prompt = buildTaskContractPrompt({
      taskId: "task-1",
      runId: "run-1",
      taskRevision: 2,
      goal: "Implement durable tasks",
      explicitCompletionTools: true,
      allowFollowUpTasks: true,
    });

    expect(prompt).toContain("conductor_child_create_followup_task");
  });

  it("builds run-scoped conductor tools for native child sessions", async () => {
    const progressCalls: unknown[] = [];
    const completeCalls: unknown[] = [];
    const gateCalls: unknown[] = [];
    const tools = buildRunScopedConductorTools({
      onConductorProgress: async (params) => {
        progressCalls.push(params);
      },
      onConductorComplete: async (params) => {
        completeCalls.push(params);
      },
      onConductorGate: async (params) => {
        gateCalls.push(params);
      },
    });

    expect(tools.map((tool) => tool.name)).toEqual([
      "conductor_child_progress",
      "conductor_child_create_gate",
      "conductor_child_complete",
    ]);

    await tools[0]?.execute?.(
      "call-1",
      { runId: "run-1", taskId: "task-1", progress: "half done" } as never,
      undefined as never,
      undefined as never,
      undefined as never,
    );
    await tools[1]?.execute?.(
      "call-2",
      {
        runId: "run-1",
        taskId: "task-1",
        type: "needs_input",
        requestedDecision: "Which database should I use?",
      } as never,
      undefined as never,
      undefined as never,
      undefined as never,
    );
    await tools[2]?.execute?.(
      "call-3",
      {
        runId: "run-1",
        taskId: "task-1",
        status: "succeeded",
        completionSummary: "done",
      } as never,
      undefined as never,
      undefined as never,
      undefined as never,
    );

    expect(progressCalls).toEqual([{ runId: "run-1", taskId: "task-1", progress: "half done" }]);
    expect(gateCalls).toEqual([
      { runId: "run-1", taskId: "task-1", type: "needs_input", requestedDecision: "Which database should I use?" },
    ]);
    expect(completeCalls).toEqual([
      { runId: "run-1", taskId: "task-1", status: "succeeded", completionSummary: "done" },
    ]);
  });

  it("adds a scoped follow-up task tool only when the task contract allows it", async () => {
    const followUpCalls: unknown[] = [];
    const tools = buildRunScopedConductorTools({
      taskContract: {
        taskId: "task-1",
        runId: "run-1",
        taskRevision: 1,
        goal: "Do it",
        explicitCompletionTools: true,
        allowFollowUpTasks: true,
      },
      onConductorFollowUpTask: async (params) => {
        followUpCalls.push(params);
      },
    });

    expect(tools.map((tool) => tool.name)).toContain("conductor_child_create_followup_task");
    const followUpTool = tools.find((tool) => tool.name === "conductor_child_create_followup_task");
    await followUpTool?.execute?.(
      "call-1",
      { runId: "run-1", taskId: "task-1", title: "Follow up", prompt: "Do the follow-up" } as never,
      undefined as never,
      undefined as never,
      undefined as never,
    );

    expect(followUpCalls).toEqual([
      { runId: "run-1", taskId: "task-1", title: "Follow up", prompt: "Do the follow-up" },
    ]);
    expect(
      buildRunScopedConductorTools({
        taskContract: {
          taskId: "task-1",
          runId: "run-1",
          taskRevision: 1,
          goal: "Do it",
          explicitCompletionTools: true,
        },
      }).map((tool) => tool.name),
    ).not.toContain("conductor_child_create_followup_task");
  });

  it("rejects scoped child tool calls for another task or run", async () => {
    const tools = buildRunScopedConductorTools({
      taskContract: {
        taskId: "task-1",
        runId: "run-1",
        taskRevision: 1,
        goal: "Do it",
        explicitCompletionTools: true,
      },
    });

    await expect(
      tools[0]?.execute?.(
        "call-1",
        { runId: "other-run", taskId: "task-1", progress: "spoofed" } as never,
        undefined as never,
        undefined as never,
        undefined as never,
      ),
    ).rejects.toThrow(/not scoped/i);
    await expect(
      tools[2]?.execute?.(
        "call-2",
        { runId: "run-1", taskId: "other-task", status: "succeeded", completionSummary: "spoofed" } as never,
        undefined as never,
        undefined as never,
        undefined as never,
      ),
    ).rejects.toThrow(/not scoped/i);
  });

  it("maps Pi stop reasons to conductor run outcomes", () => {
    for (const reason of ["pending", "deferred"] as const) {
      expect(mapStopReasonToRunOutcome(reason)).toMatchObject({ status: "error" });
    }
    expect(mapStopReasonToRunOutcome("stop")).toEqual({ status: "success", errorMessage: null });
    expect(mapStopReasonToRunOutcome("aborted")).toEqual({ status: "aborted", errorMessage: null });
    expect(mapStopReasonToRunOutcome("error")).toEqual({ status: "error", errorMessage: null });
    expect(mapStopReasonToRunOutcome("toolUse")).toEqual({
      status: "error",
      errorMessage: "Run ended unexpectedly while waiting on tool execution",
    });
    expect(mapStopReasonToRunOutcome("length")).toEqual({
      status: "error",
      errorMessage:
        "Run stopped because the model hit its output or context length limit; shorten or split the task and retry",
    });
  });

  it("validates worker context before declaring preflight success", async () => {
    vi.spyOn(ModelRuntime, "create").mockResolvedValue({
      getAvailable: () => [{ id: "fake-model" }],
    } as unknown as ModelRuntime);

    const worktreePath = mkdtempSync(join(tmpdir(), "pi-conductor-runtime-"));
    const sessionFile = join(worktreePath, "session.jsonl");
    writeFileSync(sessionFile, "{}\n", "utf-8");

    await expect(preflightWorkerRunRuntime({ worktreePath, sessionFile })).resolves.toBeUndefined();
    await expect(preflightWorkerRunRuntime({ worktreePath: "/missing", sessionFile })).rejects.toThrow(/worktree/i);
    await expect(preflightWorkerRunRuntime({ worktreePath, sessionFile: "/missing/session.jsonl" })).rejects.toThrow(
      /session file/i,
    );
  });

  it("extracts final assistant text content and falls back cleanly when absent", () => {
    expect(
      extractFinalAssistantText([
        {
          role: "assistant",
          content: [
            { type: "thinking", thinking: "..." },
            { type: "text", text: "Implemented status output." },
            { type: "text", text: "Tests are green." },
          ],
        },
      ]),
    ).toBe("Implemented status output.\n\nTests are green.");

    expect(
      extractFinalAssistantText([
        {
          role: "assistant",
          content: [{ type: "thinking", thinking: "..." }],
        },
      ]),
    ).toBeNull();
  });

  it("returns aborted status when a running prompt is cancelled", async () => {
    const worktreePath = mkdtempSync(join(tmpdir(), "pi-conductor-runtime-"));
    const sessionFile = join(worktreePath, "session.jsonl");
    writeFileSync(sessionFile, "{}\n", "utf-8");
    codingAgentMocks.openSession.mockReturnValue({} as never);

    let continuePrompt: (() => void) | null = null;
    let resolvePromptStarted: (() => void) | null = null;
    const promptStarted = new Promise<void>((resolve) => {
      resolvePromptStarted = resolve;
    });
    const session = {
      sessionId: "run-session-abort",
      messages: [] as unknown[],
      bindExtensions: vi.fn(async () => {}),
      prompt: vi.fn(async () => {
        resolvePromptStarted?.();
        await new Promise<void>((resolve) => {
          continuePrompt = resolve;
        });
        throw new Error("interrupted");
      }),
      abort: vi.fn(async () => {
        continuePrompt?.();
      }),
      dispose: vi.fn(),
    };
    codingAgentMocks.createAgentSession.mockResolvedValue({ session });

    const controller = new AbortController();
    const runtime = runWorkerPromptRuntime({ worktreePath, sessionFile, task: "do work", signal: controller.signal });
    await promptStarted;
    controller.abort();

    const result = await runtime;

    expect(result.status).toBe("aborted");
    expect(result.sessionId).toBe("run-session-abort");
    expect(session.abort).toHaveBeenCalledTimes(1);
    expect(session.dispose).toHaveBeenCalledTimes(1);
  });

  it("ignores assistant messages that were present before prompt execution", async () => {
    const worktreePath = mkdtempSync(join(tmpdir(), "pi-conductor-runtime-"));
    const sessionFile = join(worktreePath, "session.jsonl");
    writeFileSync(sessionFile, "{}\n", "utf-8");
    codingAgentMocks.openSession.mockReturnValue({} as never);

    const session = {
      sessionId: "run-session-stale",
      messages: [
        { role: "assistant", content: [{ type: "text", text: "stale pre-run summary" }], stopReason: "stop" },
      ] as unknown[],
      bindExtensions: vi.fn(async () => {}),
      prompt: vi.fn(async () => {
        session.messages.push({ role: "user", content: [{ type: "text", text: "still in progress" }] });
      }),
      abort: vi.fn(async () => {}),
      dispose: vi.fn(),
    };
    codingAgentMocks.createAgentSession.mockResolvedValue({ session });

    const result = await runWorkerPromptRuntime({ worktreePath, sessionFile, task: "do work" });

    expect(result.status).toBe("error");
    expect(result.errorMessage).toBe("Run finished without a terminal assistant message");
    expect(result.sessionId).toBe("run-session-stale");
  });

  it("fails preflight when no model provider is configured", async () => {
    vi.spyOn(ModelRuntime, "create").mockResolvedValue({ getAvailable: () => [] } as unknown as ModelRuntime);

    const worktreePath = mkdtempSync(join(tmpdir(), "pi-conductor-runtime-"));
    const sessionFile = join(worktreePath, "session.jsonl");
    writeFileSync(sessionFile, "{}\n", "utf-8");

    await expect(preflightWorkerRunRuntime({ worktreePath, sessionFile })).rejects.toThrow(
      /No usable model or provider configuration/,
    );
  });

  it("routes headless execution through a runtime backend interface", async () => {
    const worktreePath = mkdtempSync(join(tmpdir(), "pi-conductor-runtime-"));
    const sessionFile = join(worktreePath, "session.jsonl");
    writeFileSync(sessionFile, "{}\n", "utf-8");
    codingAgentMocks.openSession.mockReturnValue({} as never);

    const session = {
      sessionId: "run-session-backend",
      messages: [] as unknown[],
      bindExtensions: vi.fn(async () => {}),
      prompt: vi.fn(async () => {
        session.messages.push({ role: "assistant", stopReason: "stop", content: [{ type: "text", text: "done" }] });
      }),
      abort: vi.fn(async () => {}),
      dispose: vi.fn(),
    };
    codingAgentMocks.createAgentSession.mockResolvedValue({ session });

    const backend = getWorkerRunRuntimeBackend("headless");
    expect(backend.mode).toBe("headless");
    const result = await backend.run({ worktreePath, sessionFile, task: "do work" });

    expect(result).toMatchObject({ status: "success", finalText: "done", sessionId: "run-session-backend" });
  });

  it("uses the real Pi SDK loader, exact tools, and persisted session without provider calls", async () => {
    const actual = await vi.importActual<typeof import("@earendil-works/pi-coding-agent")>(
      "@earendil-works/pi-coding-agent",
    );
    const dir = mkdtempSync(join(tmpdir(), "pi-conductor-sdk-"));
    const fetchMock = vi.spyOn(globalThis, "fetch").mockRejectedValue(new Error("Unexpected network request"));
    try {
      const runtime = await createModelRuntime({
        authPath: join(dir, "auth.json"),
        modelsPath: null,
        refreshOnCreate: false,
      });
      const model = runtime.getModel("anthropic", "claude-sonnet-4-5");
      if (!model) throw new Error("Missing fixture model");
      vi.mocked(ModelRuntime.create).mockResolvedValue(runtime);
      vi.spyOn(runtime, "getAvailable").mockResolvedValue([model]);
      const manager = actual.SessionManager.create(dir, dir);
      const sessionFile = manager.getSessionFile();
      if (!sessionFile) throw new Error("Missing fixture session file");
      // Persist a minimal session before handing it to the worker's open/resume path.
      writeFileSync(sessionFile, `${JSON.stringify(manager.getHeader())}\n`);
      codingAgentMocks.openSession.mockImplementation(actual.SessionManager.open);
      let activeTools: string[] = [];
      codingAgentMocks.createAgentSession.mockImplementation(async (options) => {
        expect(options.modelRuntime).toBe(runtime);
        expect(options).not.toHaveProperty("authStorage");
        expect(options).not.toHaveProperty("modelRegistry");
        expect(options.resourceLoader.getSystemPromptSource()).toBeUndefined();
        expect(options.resourceLoader.getAppendSystemPromptSources()).toEqual([]);
        const result = await actual.createAgentSession({
          ...options,
          model,
          agentDir: dir,
          settingsManager: actual.SettingsManager.inMemory(),
        });
        activeTools = result.session.getActiveToolNames();
        vi.spyOn(result.session, "prompt").mockImplementation(async () => {
          const message = {
            role: "assistant" as const,
            content: [{ type: "text" as const, text: "isolated SDK success" }],
            api: model.api,
            provider: model.provider,
            model: model.id,
            usage: {
              input: 0,
              output: 0,
              cacheRead: 0,
              cacheWrite: 0,
              totalTokens: 0,
              cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
            },
            stopReason: "stop" as const,
            timestamp: Date.now(),
          };
          options.sessionManager.appendMessage(message);
          result.session.messages.push(message);
        });
        return result;
      });
      const input = { worktreePath: dir, sessionFile, task: "isolated check" };
      await expect(preflightWorkerRunRuntime(input)).resolves.toBeUndefined();
      const result = await runWorkerPromptRuntime(input);
      expect(result.errorMessage).toBeNull();
      expect(result).toMatchObject({ status: "success", sessionId: manager.getSessionId() });
      expect(activeTools.sort()).toEqual(["read", "bash", "edit", "write", "grep", "find", "ls"].sort());
      expect(actual.SessionManager.open(sessionFile).getSessionId()).toBe(manager.getSessionId());
      expect(readFileSync(sessionFile, "utf8")).toContain("isolated SDK success");
      expect(fetchMock).not.toHaveBeenCalled();
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it("returns aborted when cancellation interrupts asynchronous model initialization", async () => {
    const controller = new AbortController();
    vi.mocked(ModelRuntime.create).mockImplementation(async ({ signal } = {}) => {
      expect(signal).toBe(controller.signal);
      controller.abort();
      throw new Error("initialization aborted");
    });
    await expect(
      runWorkerPromptRuntime({
        worktreePath: "/unused",
        sessionFile: "/unused",
        task: "unused",
        signal: controller.signal,
      }),
    ).resolves.toMatchObject({ status: "aborted", sessionId: null });
    expect(codingAgentMocks.createAgentSession).not.toHaveBeenCalled();
  });

  it("does not start a prompt when cancellation arrives during session setup", async () => {
    const controller = new AbortController();
    const session = {
      sessionId: "cancelled-setup",
      messages: [],
      bindExtensions: vi.fn(async () => {
        controller.abort();
      }),
      prompt: vi.fn(),
      abort: vi.fn(async () => {}),
      dispose: vi.fn(),
    };
    codingAgentMocks.createAgentSession.mockResolvedValue({ session });
    await expect(
      runWorkerPromptRuntime({
        worktreePath: "/unused",
        sessionFile: "/unused",
        task: "unused",
        signal: controller.signal,
      }),
    ).resolves.toMatchObject({ status: "aborted", sessionId: "cancelled-setup" });
    expect(session.prompt).not.toHaveBeenCalled();
    expect(session.dispose).toHaveBeenCalledOnce();
  });

  it("does not initialize a session or auth for an already-cancelled run", async () => {
    const controller = new AbortController();
    controller.abort();
    await expect(
      runWorkerPromptRuntime({
        worktreePath: "/unused",
        sessionFile: "/unused",
        task: "unused",
        signal: controller.signal,
      }),
    ).resolves.toMatchObject({ status: "aborted", sessionId: null });
    expect(ModelRuntime.create).not.toHaveBeenCalled();
    expect(codingAgentMocks.openSession).not.toHaveBeenCalled();
    expect(codingAgentMocks.createAgentSession).not.toHaveBeenCalled();
  });

  it("maps conductor run statuses to runtime statuses", () => {
    expect(mapRunStatusToRuntimeStatus("queued")).toBe("running");
    expect(mapRunStatusToRuntimeStatus("dispatch_pending")).toBe("running");
    expect(mapRunStatusToRuntimeStatus("succeeded")).toBe("exited_success");
    expect(mapRunStatusToRuntimeStatus("partial")).toBe("exited_success");
    expect(mapRunStatusToRuntimeStatus("blocked")).toBe("exited_success");
    expect(mapRunStatusToRuntimeStatus("failed")).toBe("exited_error");
    expect(mapRunStatusToRuntimeStatus("stale")).toBe("exited_error");
    expect(mapRunStatusToRuntimeStatus("unknown_dispatch")).toBe("exited_error");
    expect(mapRunStatusToRuntimeStatus("aborted")).toBe("aborted");
    expect(mapRunStatusToRuntimeStatus("interrupted")).toBe("aborted");
    expect(mapRunStatusToRuntimeStatus(undefined)).toBe("unknown");
  });

  it("selects supervised tmux backends including the iTerm2 viewer mode", () => {
    expect(getWorkerRunRuntimeBackend("tmux")).toMatchObject({ mode: "tmux" });
    expect(getWorkerRunRuntimeBackend("iterm-tmux")).toMatchObject({ mode: "iterm-tmux" });
  });

  it("returns successful outcome from assistant final state and text extraction", async () => {
    const worktreePath = mkdtempSync(join(tmpdir(), "pi-conductor-runtime-"));
    const sessionFile = join(worktreePath, "session.jsonl");
    writeFileSync(sessionFile, "{}\n", "utf-8");
    codingAgentMocks.openSession.mockReturnValue({} as never);

    const session = {
      sessionId: "run-session-success",
      messages: [] as unknown[],
      bindExtensions: vi.fn(async () => {}),
      prompt: vi.fn(async () => {
        session.messages.push({ role: "assistant", stopReason: "stop", content: [{ type: "text", text: "done now" }] });
      }),
      abort: vi.fn(async () => {}),
      dispose: vi.fn(),
    };
    codingAgentMocks.createAgentSession.mockResolvedValue({ session });

    const result = await runWorkerPromptRuntime({ worktreePath, sessionFile, task: "do work" });

    expect(result.status).toBe("success");
    expect(result.finalText).toBe("done now");
    expect(result.errorMessage).toBeNull();
    expect(result.sessionId).toBe("run-session-success");
  });

  it("enables run-scoped conductor tools in child session allowlist", async () => {
    const worktreePath = mkdtempSync(join(tmpdir(), "pi-conductor-runtime-"));
    const sessionFile = join(worktreePath, "session.jsonl");
    writeFileSync(sessionFile, "{}\n", "utf-8");
    codingAgentMocks.openSession.mockReturnValue({} as never);

    const session = {
      sessionId: "run-session-tools",
      messages: [] as unknown[],
      bindExtensions: vi.fn(async () => {}),
      prompt: vi.fn(async () => {
        session.messages.push({ role: "assistant", stopReason: "stop", content: [{ type: "text", text: "done" }] });
      }),
      abort: vi.fn(async () => {}),
      dispose: vi.fn(),
    };
    codingAgentMocks.createAgentSession.mockResolvedValue({ session });

    await runWorkerPromptRuntime({
      worktreePath,
      sessionFile,
      task: "do work",
      taskContract: {
        taskId: "task-1",
        runId: "run-1",
        taskRevision: 1,
        goal: "do work",
        explicitCompletionTools: true,
        allowFollowUpTasks: true,
      },
    });

    expect(codingAgentMocks.createAgentSession).toHaveBeenCalledWith(
      expect.objectContaining({
        tools: [
          "read",
          "bash",
          "edit",
          "write",
          "grep",
          "find",
          "ls",
          "conductor_child_progress",
          "conductor_child_create_gate",
          "conductor_child_create_followup_task",
          "conductor_child_complete",
        ],
        customTools: expect.arrayContaining([
          expect.objectContaining({ name: "conductor_child_progress" }),
          expect.objectContaining({ name: "conductor_child_create_gate" }),
          expect.objectContaining({ name: "conductor_child_create_followup_task" }),
          expect.objectContaining({ name: "conductor_child_complete" }),
        ]),
      }),
    );
  });
});

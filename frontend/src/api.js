// Thin fetch wrapper for the GradeForge API (proxied to FastAPI by Vite in development).

export class ApiError extends Error {
  constructor(status, detail) {
    super(typeof detail === "string" ? detail : JSON.stringify(detail));
    this.status = status;
    this.detail = detail;
  }
}

async function request(path, { method = "GET", body, form } = {}) {
  const options = { method };
  if (form) options.body = form;
  else if (body !== undefined) {
    options.body = JSON.stringify(body);
    options.headers = { "Content-Type": "application/json" };
  }
  const response = await fetch(`/api${path}`, options);
  const data = response.headers.get("content-type")?.includes("json") ? await response.json() : null;
  if (!response.ok) throw new ApiError(response.status, data?.detail ?? response.statusText);
  return data;
}

const upload = (path, fields) => {
  const form = new FormData();
  Object.entries(fields).forEach(([k, v]) => form.append(k, v));
  return request(path, { method: "POST", form });
};

export const api = {
  health: () => request("/health"),
  settings: () => request("/settings"),
  saveSettings: (changes) => request("/settings", { method: "PUT", body: changes }),
  job: (id) => request(`/jobs/${id}`),

  models: () => request("/models"),
  modelCard: (name) => request(`/models/card?${new URLSearchParams({ name })}`),
  gpu: () => request("/models/gpu"),
  probeModel: (name) => request("/models/probe", { method: "POST", body: { name } }),
  resolveModel: (name) => request("/models/resolve", { method: "POST", body: { name } }),
  setRepo: (name, repo) => request("/models/resolution", { method: "PUT", body: { name, repo } }),
  switchModel: (name) => request("/models/switch", { method: "POST", body: { name } }),
  pullModel: (name) => request("/models/pull", { method: "POST", body: { name } }),
  importModel: (body) => request("/models/import", { method: "POST", body }),
  hfToken: () => request("/models/hf-token"),
  setHfToken: (token) => request("/models/hf-token", { method: "PUT", body: { token } }),
  deleteHfToken: () => request("/models/hf-token", { method: "DELETE" }),
  storage: (keep = 2) => request(`/storage?keep=${keep}`),
  cleanStorage: (keep = 2) => request("/storage/clean", { method: "POST", body: { keep } }),

  cloud: () => request("/cloud"),
  cloudModelInfo: (model) => request(`/cloud/model-info?${new URLSearchParams({ model })}`),
  saveCloudKey: (provider, key) => request("/cloud/key", { method: "PUT", body: { provider, key } }),
  deleteCloudKey: (provider) => request(`/cloud/key/${provider}`, { method: "DELETE" }),
  testCloudKey: (provider, model) => request("/cloud/test", { method: "POST", body: { provider, model } }),
  cloudEstimate: (model, papers) => request("/cloud/estimate", { method: "POST", body: { model, papers } }),
  saveCloudSettings: (body) => request("/cloud/settings", { method: "PUT", body }),

  exams: () => request("/exams"),
  createDemo: () => request("/demo", { method: "POST" }),
  createExam: (name, subject) => request("/exams", { method: "POST", body: { name, subject } }),
  exam: (id) => request(`/exams/${id}`),
  analytics: (id) => request(`/exams/${id}/analytics`),
  uploadPaper: (id, file) => upload(`/exams/${id}/paper`, { file }),
  saveQuestions: (id, questions) => request(`/exams/${id}/questions`, { method: "PUT", body: questions }),

  generateKey: (id) => request(`/exams/${id}/answer-key/generate`, { method: "POST" }),
  saveKey: (id, key) => request(`/exams/${id}/answer-key`, { method: "PUT", body: key }),
  validateKey: (id) => request(`/exams/${id}/answer-key/validate`, { method: "POST" }),
  finalizeKey: (id) => request(`/exams/${id}/answer-key/finalize`, { method: "POST" }),
  discuss: (id, qid, message) =>
    request(`/exams/${id}/disputes/${qid}/discuss`, { method: "POST", body: { message } }),
  resolve: (id, qid, body) => request(`/exams/${id}/disputes/${qid}/resolve`, { method: "POST", body }),

  uploadSheet: (id, student, file) => upload(`/exams/${id}/sheets`, { student, file }),
  sheet: (id, sid) => request(`/exams/${id}/sheets/${sid}`),
  fixLine: (id, sid, page, index, text) =>
    request(`/exams/${id}/sheets/${sid}/lines/${page}/${index}`, { method: "PUT", body: { text } }),
  grade: (id, sid, strictness) =>
    request(`/exams/${id}/sheets/${sid}/grade`, { method: "POST", body: { strictness } }),
  whatIf: (id, sid, strictness, save = false) =>
    request(`/exams/${id}/sheets/${sid}/what-if`, { method: "POST", body: { strictness, save } }),
  fixMarks: (id, sid, qid, body) =>
    request(`/exams/${id}/sheets/${sid}/questions/${qid}`, { method: "PUT", body }),
  approveSheet: (id, sid, teacher_name) =>
    request(`/exams/${id}/sheets/${sid}/approve`, { method: "POST", body: { teacher_name } }),

  disputes: (params) => request(`/disputes?${new URLSearchParams(params)}`),

  learning: () => request("/learning/status"),
  trainTrocr: (writer) => request("/learning/trocr/train", { method: "POST", body: { writer } }),
  llmPlan: (model) => request(`/learning/llm/plan?${new URLSearchParams(model ? { model } : {})}`),
  progress: () => request("/learning/progress"),
  benchmark: (setups) => request("/learning/benchmark", { method: "POST", body: setups ? { setups } : {} }),
  weights: (model, checkSize = false) =>
    request(`/learning/llm/weights?${new URLSearchParams({ ...(model ? { model } : {}), check_size: checkSize })}`),
  downloadWeights: (model) => request("/learning/llm/weights/download", { method: "POST", body: { model } }),
  retrain: (model, fromScratch = false) =>
    request("/learning/llm/retrain", { method: "POST", body: { model, from_scratch: fromScratch } }),
  pauseTraining: () => request("/learning/llm/pause", { method: "POST" }),
  importAdapter: (model, path) => request("/learning/llm/import-adapter", { method: "POST", body: { model, path } }),
  llmVersions: (model) => request(`/learning/llm/versions?${new URLSearchParams(model ? { model } : {})}`),
  exportNotebook: async (model) => {
    const response = await fetch("/api/learning/llm/export", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ consent: true, model }),
    });
    if (!response.ok) throw new ApiError(response.status, (await response.json()).detail);
    const link = document.createElement("a");
    link.href = URL.createObjectURL(await response.blob());
    link.download = "gradeforge_finetune.ipynb";
    link.click();
    URL.revokeObjectURL(link.href);
  },
};

export const formatBytes = (n) => {
  if (n == null) return "";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  for (; n >= 1024 && i < units.length - 1; i++) n /= 1024;
  return `${n.toFixed(n >= 10 || i === 0 ? 0 : 1)} ${units[i]}`;
};

export const assetUrl =(examId, sheetId, kind, name) => `/api/exams/${examId}/sheets/${sheetId}/${kind}/${name}`;

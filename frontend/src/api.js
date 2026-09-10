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
  models: () => request("/models"),
  job: (id) => request(`/jobs/${id}`),

  exams: () => request("/exams"),
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

  disputes: (params) => request(`/disputes?${new URLSearchParams(params)}`),
};

export const assetUrl = (examId, sheetId, kind, name) => `/api/exams/${examId}/sheets/${sheetId}/${kind}/${name}`;

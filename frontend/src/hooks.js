import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { api } from "./api";

// Poll a background job until it finishes; then invalidate the given query keys.
export function useJob(onDone) {
  const [jobId, setJobId] = useState(null);
  const queryClient = useQueryClient();
  const query = useQuery({
    queryKey: ["job", jobId],
    queryFn: () => api.job(jobId),
    enabled: !!jobId,
    refetchInterval: (q) => (["done", "failed"].includes(q.state.data?.status) ? false : 700),
  });
  const job = query.data;

  useEffect(() => {
    if (job?.status === "done" || job?.status === "failed") onDone?.(job, queryClient);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [job?.status]);

  return {
    job,
    running: !!jobId && !["done", "failed"].includes(job?.status),
    start: (id) => setJobId(id),
    clear: () => setJobId(null),
  };
}

export function useSettings() {
  return useQuery({ queryKey: ["settings"], queryFn: api.settings });
}

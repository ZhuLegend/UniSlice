/* Pure aggregation shared by the offline report and its regression tests. */
(function (root) {
  function aggregate(data, windowSeconds, thresholds, sampleRates) {
    const n = Math.max(1, Math.ceil(data.duration / windowSeconds));
    const series = data.models.map(() => Array.from({length:n}, (_,i) => ({
      start:i*windowSeconds, end:Math.min((i+1)*windowSeconds,data.duration),
      arrivals:0, completed:0, hits:0, eligible:0, unresolved:0, failed:0, arrivedRequests:0, terminalRequests:0, failedTerminals:0, latencies:[]
    })));
    const totals = data.models.map(() => ({arrivals:0,completed:0,samples:0,failed:0,pending:0,hits:0,eligible:0,latencies:[]}));
    for (const [m, arrival, completion, batch, status] of data.tasks) {
      const bucket = series[m][Math.min(n-1, Math.floor(arrival/windowSeconds))];
      const total = totals[m];
      bucket.arrivals += sampleRates ? batch : 1;
      bucket.arrivedRequests++;
      total.arrivals++;
      if (status !== 0 && completion !== null && completion >= 0 && completion <= data.duration) {
        const terminal = series[m][Math.min(n-1,Math.floor(completion/windowSeconds))];
        terminal.terminalRequests++;
        if (status === 2) terminal.failedTerminals += sampleRates ? batch : 1;
      }
      const threshold = thresholds[m]/1000;
      const observed = status !== 0 || data.duration-arrival >= threshold;
      if (observed) {bucket.eligible++; total.eligible++;}
      else bucket.unresolved++;
      if (status === 1) {
        total.completed++; total.samples+=batch;
        const latency=(completion-arrival)*1000;
        total.latencies.push(latency); bucket.latencies.push(latency);
        if (latency <= thresholds[m]+1e-6) {bucket.hits++; total.hits++;}
        if (completion >= 0 && completion <= data.duration) {
          series[m][Math.min(n-1,Math.floor(completion/windowSeconds))].completed += sampleRates ? batch : 1;
        }
      } else if (status === 2) {bucket.failed++; total.failed++;}
      else total.pending++;
    }
    for (const s of series) {
     let inFlight=0;
     for (const b of s) {
      const width=b.end-b.start;
      b.arrivalRate=b.arrivals/width;
      b.completionRate=b.completed/width;
      b.failureRate=b.failedTerminals/width;
      inFlight+=b.arrivedRequests-b.terminalRequests;
      b.inFlight=inFlight;
      b.slo=b.eligible ? b.hits/b.eligible*100 : null;
     }
    }
    return {series,totals};
  }
  function percentile(values, q) {
    if (!values.length) return null;
    const sorted=[...values].sort((a,b)=>a-b);
    return sorted[Math.max(0,Math.ceil(q*sorted.length)-1)];
  }
  function lifecycleForCycle(data, cycle) {
    const result={loads:0,unloads:0,load_seconds:0,unload_seconds:0,drain_seconds:0,queue_gaps:0};
    for(const event of data.lifecycle||[]) {
      // Cycle boundaries use integer milliseconds; node log timestamps have
      // finer precision. Include the final millisecond so a finished operation
      // is not counted as zero while its duration still overlaps this cycle.
      if(event.time>=cycle.start && event.time<cycle.end+0.001) {
        if(event.event==='model_load_finished')result.loads++;
        if(event.event==='model_unload_finished')result.unloads++;
        if(event.event==='model_queue_without_worker')result.queue_gaps++;
      }
      const field={'model_load_finished':'load_seconds','model_unload_finished':'unload_seconds','worker_drain_finished':'drain_seconds'}[event.event];
      if(field && Number.isFinite(event.elapsed_seconds)) {
        result[field]+=Math.max(0,Math.min(event.time,cycle.end)-Math.max(event.time-event.elapsed_seconds,cycle.start));
      }
    }
    return result;
  }
  const api={aggregate,percentile,lifecycleForCycle};
  if (typeof module !== 'undefined' && module.exports) module.exports=api;
  else root.TimelineMetrics=api;
})(globalThis);

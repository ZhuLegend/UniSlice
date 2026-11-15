const assert=require('node:assert/strict');
const {aggregate}=require('./timeline_metrics.js');
const data={duration:2.5,models:['a','b'],tasks:[
  [0,0.8,1.1,3,1], // completion belongs to the next time window; arrival-cohort SLO fails
  [0,1.2,1.3,2,1], // on-time success
  [0,1.4,1.5,1,2], // failed requests must be in the denominator
  [0,2.45,null,1,0], // threshold not yet elapsed: unresolved, not a miss
  [1,1.1,null,1,0] // pending beyond deadline: miss
]};
const r=aggregate(data,1,[200,200],false);
assert.equal(r.series[0][0].arrivalRate,1);
assert.equal(r.series[0][0].completionRate,0);
assert.equal(r.series[0][0].slo,0);
assert.equal(r.series[0][1].completionRate,2);
assert.equal(r.series[0][1].slo,50);
assert.equal(r.series[0][2].arrivalRate,2); // partial final window is 0.5 seconds
assert.equal(r.series[0][2].slo,null);
assert.equal(r.series[0][2].unresolved,1);
assert.equal(r.series[1][0].slo,null); // no arrivals is not 100% attainment
assert.equal(r.series[1][1].slo,0);
assert.equal(r.totals[0].hits,1);
assert.equal(r.totals[0].eligible,3);
assert.equal(aggregate(data,1,[200,200],true).series[0][1].completionRate,5);
assert.equal(aggregate(data,1,[400,200],false).totals[0].hits,2);
console.log('Timeline metrics: 13 assertions passed');

assert.equal(r.series[0][0].inFlight,1);
assert.equal(r.series[0][1].inFlight,0); // Success and explicit failure both finish requests.
assert.equal(r.series[0][2].inFlight,1);
assert.equal(r.series[1][1].inFlight,1); // Pending is retained until an actual terminal event.
assert.equal(r.series[0][0].failureRate,0);
assert.equal(r.series[0][1].failureRate,1);
const deferredFailure=aggregate({duration:3,models:['m'],tasks:[[0,0.1,2.1,5,2]]},1,[250],true);
assert.deepEqual(deferredFailure.series[0].map(b=>b.inFlight),[1,1,0]);
assert.deepEqual(deferredFailure.series[0].map(b=>b.failureRate),[0,0,5]);
console.log('In-flight and failure timeline: 8 assertions passed');

const {lifecycleForCycle}=require('./timeline_metrics.js');
const costs=lifecycleForCycle({lifecycle:[
 {event:'model_load_finished',time:4,elapsed_seconds:2},
 {event:'model_load_finished',time:3.5,elapsed_seconds:2},
 {event:'model_unload_finished',time:3.2,elapsed_seconds:0.1},
 {event:'worker_drain_finished',time:3.1,elapsed_seconds:0.2},
 {event:'model_queue_without_worker',time:3.2},
 {event:'model_load_started',time:3.3}
]}, {start:3,end:4});
assert.equal(costs.loads,2);
assert.equal(costs.unloads,1);
assert.equal(costs.load_seconds,1.5); // Parallel worker time, clipped at interval boundaries.
assert.ok(Math.abs(costs.unload_seconds-0.1)<1e-9);
assert.ok(Math.abs(costs.drain_seconds-0.1)<1e-9);
assert.equal(costs.queue_gaps,1);
console.log('Lifecycle metrics: 6 assertions passed');

const boundary=lifecycleForCycle({lifecycle:[
 {event:'model_load_finished',time:4.0008,elapsed_seconds:1},
 {event:'model_unload_finished',time:4.002,elapsed_seconds:0.1}
]}, {start:3,end:4});
assert.equal(boundary.loads,1); // Same final millisecond as the integer cycle end.
assert.equal(boundary.unloads,0); // A later completion remains outside this cycle.
console.log('Lifecycle timestamp boundary: 2 assertions passed');

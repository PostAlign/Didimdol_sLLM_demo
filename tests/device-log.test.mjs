import test from 'node:test';
import assert from 'node:assert/strict';
import { parseDeviceReport, parseReportDate, parseFileDate, matchDeviceReports, deviceLogFromKill, reportCoverage, coverageLabel, WEB_CONTENT } from '../web/sllm/experiments/device-log.js';
import { parseDeviceLogNote } from '../web/sllm/experiments/results.js';

// Trimmed from the September 13 phone reports: three processes, real page counts.
const jetsam = (date, processes, free = 4272) => JSON.stringify({ bug_type: '298', timestamp: `${date.slice(0, 19)}.00 +0900` }) + '\n' +
  JSON.stringify({ build: 'iPhone OS 26.6.2 (23G90)', product: 'iPhone15,3', date, bug_type: '298', largestProcess: WEB_CONTENT,
    memoryStatus: { compressorSize: 73615, pageSize: 16384, memoryPages: { free, anonymous: 157702 } },
    processes: [{ name: 'kernel_task', pid: 0, rpages: 14665, states: ['active'] }, ...processes] });
const webContent = (pid, rpages, extra = {}) => ({ name: WEB_CONTENT, pid, rpages, lifetimeMax: 117805, states: ['active'], cpuTime: 38.6, ...extra });

const firstReport = jetsam('2026-09-13 17:39:18.43 +0900', [webContent(2023, 36086),
  { name: 'SharingUIService', pid: 1843, rpages: 1425, reason: 'highwater', states: ['suspended'] }]);
const secondReport = jetsam('2026-09-13 17:40:06.65 +0900', [
  webContent(2023, 112268, { reason: 'highwater' }), webContent(2076, 146146, { reason: 'highwater', lifetimeMax: 146146 })], 10850);
const KST = -540;
const kst = text => Date.parse(text.replace(' ', 'T') + '+09:00');

test('report dates and file names resolve to the same instant on the device clock', () => {
  assert.equal(parseReportDate('2026-09-13 17:40:06.65 +0900'), kst('2026-09-13 17:40:06.65'));
  assert.equal(parseReportDate('2026-09-13 17:40:06.00 +0900'), kst('2026-09-13 17:40:06'));
  assert.equal(parseReportDate('garbage'), null);
  assert.equal(parseFileDate('JetsamEvent-2026-09-13-174006.ips', KST), kst('2026-09-13 17:40:06'));
  assert.equal(parseFileDate('JetsamEvent-2026-09-13-174006.ips', null), null, 'a file name alone has no zone');
});

test('a JetsamEvent report yields the killed processes, WebContent footprints and system memory', () => {
  const report = parseDeviceReport(secondReport, 'JetsamEvent-2026-09-13-174006.ips', { timezoneOffsetMinutes: KST });
  assert.equal(report.kind, 'jetsam');
  assert.equal(report.reportedAt, kst('2026-09-13 17:40:06.65'));
  assert.equal(report.largestProcess, WEB_CONTENT);
  assert.equal(report.freeMiB, 169.5);
  assert.equal(report.compressorMiB, 1150.2);
  assert.deepEqual(report.kills.map(kill => [kill.pid, kill.reason, kill.footprintMiB, kill.lifetimeMaxMiB]),
    [[2023, 'highwater', 1754.2, 1840.7], [2076, 'highwater', 2283.5, 2283.5]]);
  assert.equal(report.webContent.length, 2);
  const quiet = parseDeviceReport(firstReport, 'JetsamEvent-2026-09-13-173918.ips');
  assert.deepEqual(quiet.kills.map(kill => kill.name), ['SharingUIService']);
  assert.equal(quiet.webContent[0].footprintMiB, 563.8, 'the surviving WebContent process is still reported');
  assert.equal(quiet.freeMiB, 66.8);
});

test('resource, crash and unreadable files are classified instead of treated as kills', () => {
  const resource = parseDeviceReport(JSON.stringify({ app_name: 'com.apple.WebKit.Networking', bug_type: '145', timestamp: '2026-09-13 17:27:12.00 +0900' }) +
    '\nDate/Time: 2026-09-13 15:57:11.972 +0900\nEvent:            disk writes\nAction taken:     none\nWrites:           4294.97 MB of file backed memory dirtied over 5399 seconds\n',
    'com.apple.WebKit.Networking.diskwrites_resource-2026-09-13-172712.ips');
  assert.equal(resource.kind, 'resource');
  assert.equal(resource.event, 'disk writes');
  assert.equal(resource.action, 'none');
  assert.match(resource.writes, /4294\.97 MB/);
  assert.equal(resource.reportedAt, kst('2026-09-13 17:27:12'));
  const crash = parseDeviceReport(JSON.stringify({ app_name: WEB_CONTENT, bug_type: '309', timestamp: '2026-09-13 17:06:53.00 +0900' }) + '\n' +
    JSON.stringify({ procName: WEB_CONTENT, pid: 1999, captureTime: '2026-09-13 17:06:53.90 +0900', termination: { indicator: 'Memory limit exceeded' } }), 'WebContent.ips');
  assert.equal(crash.kind, 'crash');
  assert.deepEqual(crash.kills.map(kill => [kill.pid, kill.reason, kill.footprintMiB]), [[1999, 'Memory limit exceeded', null]]);
  assert.equal(crash.reportedAt, kst('2026-09-13 17:06:53.90'));
  assert.equal(parseDeviceReport('not a report', 'notes.txt').kind, 'unknown');
  assert.equal(parseDeviceReport(JSON.stringify({ bug_type: '298' }) + '\nbroken', 'JetsamEvent-x.ips').error, 'JetsamEvent body is not JSON');
});

test('one report with two WebContent kills claims the two most recent interrupted rows in order', () => {
  const results = [
    { runId: 'ok', kind: 'warm-load', interrupted: false, endedAt: kst('2026-09-13 17:39:09.314') },
    { runId: 'load', kind: 'warm-load', interrupted: true, endedAt: kst('2026-09-13 17:39:23.760'), lastRecordAt: kst('2026-09-13 17:39:22.749'),
      comparison: { gpuRequestedCurrent: 985867680 } },
    { runId: 'probe', kind: 'probe', interrupted: true, endedAt: kst('2026-09-13 17:40:06.952') },
    { runId: 'streamed', kind: 'probe', interrupted: false, endedAt: kst('2026-09-13 17:41:24.367') },
  ];
  const reports = [secondReport, firstReport].map((text, i) => parseDeviceReport(text, i ? 'JetsamEvent-2026-09-13-173918.ips' : 'JetsamEvent-2026-09-13-174006.ips'));
  const { matched, unmatched, context, ignored } = matchDeviceReports(results, reports, { now: 5 });
  assert.deepEqual(matched.map(({ index, runId, deviceLog }) => [index, runId, deviceLog.pid, deviceLog.footprintMiB]),
    [[1, 'load', 2023, 1754.2], [2, 'probe', 2076, 2283.5]]);
  assert.equal(unmatched.length, 0);
  assert.equal(ignored.length, 0);
  assert.equal(context.length, 1, 'the earlier report killed no WebContent process and is reported as context');
  assert.deepEqual(context[0].kills, ['SharingUIService highwater']);
  assert.equal(context[0].webContent[0].footprintMiB, 563.8);
  const log = matched[0].deviceLog;
  assert.equal(log.note, 'JetsamEvent-2026-09-13-174006.ips · highwater · 1,754.2 MiB');
  assert.equal(log.matchedBy, 'report');
  assert.equal(log.recordedAt, 5);
  assert.equal(log.freeMiB, 169.5);
  assert.equal(log.lifetimeMaxMiB, 1840.7);
  // The generated note round-trips through the hand-written parser used by older rows.
  assert.deepEqual(parseDeviceLogNote(log.note), { note: log.note, file: log.file, reason: 'highwater', footprintMiB: 1754.2 });
});

test('kills without a row in their window stay unmatched and rows outside the tolerance are left alone', () => {
  const report = parseDeviceReport(secondReport, 'JetsamEvent-2026-09-13-174006.ips');
  const late = { runId: 'late', interrupted: true, endedAt: kst('2026-09-13 17:45:00') };
  const one = { runId: 'one', interrupted: true, endedAt: kst('2026-09-13 17:40:06.952') };
  const { matched, unmatched } = matchDeviceReports([late, one], [report]);
  assert.deepEqual(matched.map(entry => [entry.runId, entry.deviceLog.pid]), [['one', 2076]], 'a single row takes the later kill');
  assert.deepEqual(unmatched.map(kill => kill.pid), [2023]);
  assert.deepEqual(matchDeviceReports([], [report]).unmatched.map(kill => kill.pid), [2023, 2076]);
  assert.equal(matchDeviceReports([one], [{ kind: 'resource', file: 'r.ips', kills: [], webContent: [] }]).ignored[0].file, 'r.ips');
});

test('kill notes without a footprint keep the file and reason only', () => {
  const log = deviceLogFromKill({ file: 'WebContent.ips', reportedAt: 1 }, { pid: 3, reason: 'Memory limit exceeded', footprintMiB: null, lifetimeMaxMiB: null }, 2);
  assert.equal(log.note, 'WebContent.ips · Memory limit exceeded');
  assert.equal(log.footprintMiB, null);
});

// Trimmed from JetsamEvent-2026-09-13-225106.ips: Toss in front, no browser
// process at all, a telemetry daemon killed for its own limit, 156 MiB free.
const preRunReport = JSON.stringify({ bug_type: '298', timestamp: '2026-09-13 22:51:06.00 +0900' }) + '\n' +
  JSON.stringify({ build: 'iPhone OS 26.6.2 (23G90)', product: 'iPhone15,3', date: '2026-09-13 22:51:06.93 +0900', bug_type: '298', largestProcess: WEB_CONTENT,
    memoryStatus: { compressorSize: 67145, pageSize: 16384, memoryPages: { free: 9989, anonymous: 101007 } },
    processes: [
      { name: 'kernel_task', pid: 0, rpages: 15832, states: ['active'] },
      { name: WEB_CONTENT, pid: 2766, rpages: 37587, lifetimeMax: 45017, states: ['active'], coalition: 1302 },
      { name: 'Toss', pid: 2233, rpages: 11658, lifetimeMax: 14159, states: ['active', 'frontmost'], coalition: 1302 },
      { name: 'Slack', pid: 1139, rpages: 16090, states: ['suspended'] },
      { name: 'KakaoTalk', pid: 1627, rpages: 14371, states: ['suspended'] },
      { name: 'Runner', pid: 1758, rpages: 12499, states: ['suspended'] },
      { name: 'Korail', pid: 2515, rpages: 11164, states: ['suspended'] },
      { name: 'Gmail', pid: 1654, rpages: 7733, states: ['suspended'] },
      { name: 'Podcasts', pid: 677, rpages: 6027, states: ['suspended'] },
      { name: 'CloudTelemetryService', pid: 2088, rpages: 432, lifetimeMax: 458, states: ['daemon', 'idle'], reason: 'per-process-limit' },
    ] });

test('a report from before the first run is context with system pressure, never a match', () => {
  const report = parseDeviceReport(preRunReport, 'JetsamEvent-2026-09-13-225106.ips', { timezoneOffsetMinutes: KST });
  assert.equal(report.kind, 'jetsam');
  assert.deepEqual(report.kills.map(kill => [kill.name, kill.reason]), [['CloudTelemetryService', 'per-process-limit']]);
  assert.equal(report.freeMiB, 156.1);
  assert.equal(report.compressorMiB, 1049.1);
  assert.equal(report.suspendedMiB, 1060.7, 'suspended third-party apps are summed');
  assert.deepEqual(report.suspendedTop.map(process => process.name), ['Slack', 'KakaoTalk', 'Runner', 'Korail', 'Gmail']);
  assert.deepEqual(report.frontmost, ['Toss']);
  const results = [
    { runId: 'load', kind: 'load', interrupted: false, startedAt: kst('2026-09-13 22:54:44.823'), endedAt: kst('2026-09-13 22:54:55.660') },
    { runId: 'warm', kind: 'warm-load', interrupted: true, startedAt: kst('2026-09-13 23:13:04.170'), endedAt: kst('2026-09-13 23:13:15.209') },
  ];
  const coverage = reportCoverage(results, report);
  assert.equal(coverage.coverage, 'before-runs');
  assert.equal(Math.round(coverage.offsetFromFirstRunMs / 1000), -218);
  assert.match(coverageLabel(coverage), /첫 실행 3\.6분 전/);
  const { matched, unmatched, context } = matchDeviceReports(results, [report]);
  assert.equal(matched.length, 0);
  assert.equal(unmatched.length, 0);
  assert.equal(context.length, 1);
  assert.equal(context[0].coverage, 'before-runs');
  assert.equal(context[0].suspendedMiB, 1060.7);
  assert.deepEqual(context[0].frontmost, ['Toss']);
  assert.equal(context[0].webContent[0].pid, 2766, 'another app\'s WebContent is still listed; the browser had not started');
  const within = reportCoverage(results, { reportedAt: kst('2026-09-13 23:13:15') });
  assert.equal(within.coverage, 'within-runs');
  assert.equal(coverageLabel(within), '실행 구간 안');
  assert.equal(reportCoverage(results, { reportedAt: kst('2026-09-13 23:40:00') }).coverage, 'after-runs');
  assert.equal(reportCoverage([], report).coverage, 'unknown');
  assert.equal(reportCoverage(results, { reportedAt: null }).coverage, 'unknown');
  assert.equal(coverageLabel({ coverage: 'unknown' }), '실행 구간 판단 불가');
});

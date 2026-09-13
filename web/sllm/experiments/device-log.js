// Device reports from the phone's Analytics Data list, read so that interrupted
// rows can carry the OS's own account of the termination. A JetsamEvent file is
// one header line of JSON followed by a JSON body; the processes that were
// killed carry a `reason`. One report can describe several kills: the
// September 13 17:40:06 report held both the 17:39:23 kill of the previous
// WebContent process and the 17:40:06 kill of its replacement, so kills are
// matched to interrupted rows in order rather than one report per row.
// WebContent crash files (WebKit's own memory-limit termination) and resource
// reports (bug_type 145) are recognised so they are not silently ignored.
export const PAGE_BYTES = 16384;
export const WEB_CONTENT = 'com.apple.WebKit.WebContent';

const REPORT_DATE = /(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}:\d{2})(\.\d+)?\s*([+-]\d{2}):?(\d{2})?/;
const FILE_DATE = /(\d{4})-(\d{2})-(\d{2})-(\d{2})(\d{2})(\d{2})/;
const mib = (pages, pageSize = PAGE_BYTES) => Number.isFinite(pages) ? Math.round(pages * pageSize / 2 ** 20 * 10) / 10 : null;

/** "2026-09-13 17:40:06.65 +0900" (report bodies and headers) as epoch milliseconds. */
export function parseReportDate(text) {
  const match = REPORT_DATE.exec(String(text ?? ''));
  if (!match) return null;
  const value = Date.parse(`${match[1]}T${match[2]}${match[3] || ''}${match[4]}:${match[5] || '00'}`);
  return Number.isFinite(value) ? value : null;
}

/** File names carry the device's local clock without a zone; the export's `deviceClock` supplies the offset. */
export function parseFileDate(fileName, timezoneOffsetMinutes) {
  const match = FILE_DATE.exec(String(fileName ?? ''));
  if (!match || !Number.isFinite(timezoneOffsetMinutes)) return null;
  const [, year, month, day, hour, minute, second] = match.map(Number);
  return Date.UTC(year, month - 1, day, hour, minute, second) + timezoneOffsetMinutes * 60000;
}

function parseJSON(text) {
  try { const value = JSON.parse(text); return value && typeof value === 'object' ? value : null; }
  catch { return null; }
}

function processEntry(process, pageSize) {
  return { pid: process.pid ?? null, name: process.name ?? null, reason: process.reason ?? null,
    rpages: process.rpages ?? null, footprintMiB: mib(process.rpages, pageSize),
    lifetimeMaxMiB: mib(process.lifetimeMax, pageSize), states: process.states ?? null, cpuTime: process.cpuTime ?? null };
}

/** One `.ips` file as the facts the results table needs. Never throws; unreadable files come back as `kind: 'unknown'`. */
export function parseDeviceReport(text, fileName = '', { timezoneOffsetMinutes = null } = {}) {
  const source = String(text ?? '');
  const newline = source.indexOf('\n');
  const header = parseJSON(newline < 0 ? source : source.slice(0, newline)) || {};
  const bodyText = newline < 0 ? '' : source.slice(newline + 1);
  const file = fileName || header.name || null;
  const bugType = header.bug_type == null ? null : String(header.bug_type);
  const base = { file, bugType, process: header.app_name ?? header.name ?? null,
    reportedAt: parseReportDate(header.timestamp) ?? parseFileDate(file, timezoneOffsetMinutes),
    reportedAtText: header.timestamp ?? null, kills: [], webContent: [] };
  if (bugType === '298') {
    const body = parseJSON(bodyText);
    if (!body) return { ...base, kind: 'unknown', error: 'JetsamEvent body is not JSON' };
    const pageSize = body.memoryStatus?.pageSize || PAGE_BYTES;
    const processes = Array.isArray(body.processes) ? body.processes.map(process => processEntry(process, pageSize)) : [];
    const pages = body.memoryStatus?.memoryPages || {};
    return { ...base, kind: 'jetsam', reportedAt: parseReportDate(body.date) ?? base.reportedAt, reportedAtText: body.date ?? base.reportedAtText,
      largestProcess: body.largestProcess ?? null, pageSize,
      freeMiB: mib(pages.free, pageSize), compressorMiB: mib(body.memoryStatus?.compressorSize, pageSize),
      kills: processes.filter(process => process.reason),
      webContent: processes.filter(process => process.name === WEB_CONTENT) };
  }
  if (bugType === '309' || /WebContent/.test(base.process || '')) {
    const body = parseJSON(bodyText) || {};
    const isWebContent = (body.procName ?? base.process) === WEB_CONTENT;
    const reason = body.termination?.indicator ?? body.termination?.reason ?? body.exception?.type ?? 'crash';
    const kill = { pid: body.pid ?? null, name: body.procName ?? base.process, reason: String(reason), rpages: null,
      footprintMiB: null, lifetimeMaxMiB: null, states: null, cpuTime: null };
    return { ...base, kind: 'crash', reportedAt: parseReportDate(body.captureTime) ?? base.reportedAt,
      kills: isWebContent ? [kill] : [], webContent: isWebContent ? [kill] : [] };
  }
  if (bugType === '145') {
    const event = /Event:\s*(.+)/.exec(bodyText)?.[1]?.trim() ?? null;
    const action = /Action taken:\s*(.+)/.exec(bodyText)?.[1]?.trim() ?? null;
    const writes = /Writes:\s*(.+)/.exec(bodyText)?.[1]?.trim() ?? null;
    return { ...base, kind: 'resource', event, action, writes };
  }
  return { ...base, kind: 'unknown' };
}

/** The stored note for one kill, in the shape `parseDeviceLogNote` produces so hand-written and file-derived notes render alike. */
export function deviceLogFromKill(report, kill, now = Date.now()) {
  const footprint = kill.footprintMiB == null ? null : `${kill.footprintMiB.toLocaleString('en-US')} MiB`;
  return { note: [report.file, kill.reason, footprint].filter(Boolean).join(' · '),
    file: report.file, reason: kill.reason, footprintMiB: kill.footprintMiB, lifetimeMaxMiB: kill.lifetimeMaxMiB,
    pid: kill.pid, reportedAt: report.reportedAt, reportedAtText: report.reportedAtText ?? null,
    freeMiB: report.freeMiB ?? null, compressorMiB: report.compressorMiB ?? null, largestProcess: report.largestProcess ?? null,
    matchedBy: 'report', recordedAt: now };
}

/**
 * Pair WebContent kills with interrupted rows. Reports are taken in time order;
 * each report's kills belong to rows that ended after the previous report and
 * no later than the report time plus the tolerance. Kills are ordered by pid
 * (the older process died first) and aligned with the latest candidate rows,
 * so a report that describes two kills claims the two most recent rows.
 */
export function matchDeviceReports(results, reports, { toleranceMs = 60000, now = Date.now() } = {}) {
  const usable = reports.filter(report => ['jetsam', 'crash'].includes(report.kind) && Number.isFinite(report.reportedAt))
    .sort((a, b) => a.reportedAt - b.reportedAt);
  const rows = results.map((result, index) => ({ index, result, endedAt: result.endedAt ?? result.lastRecordAt ?? null }))
    .filter(row => row.result.interrupted && Number.isFinite(row.endedAt)).sort((a, b) => a.endedAt - b.endedAt);
  const taken = new Set(), matched = [], unmatched = [], context = [];
  let previousAt = -Infinity;
  for (const report of usable) {
    const kills = report.kills.filter(kill => kill.name === WEB_CONTENT).sort((a, b) => (a.pid ?? 0) - (b.pid ?? 0));
    if (!kills.length) { context.push({ file: report.file, kind: report.kind, reportedAt: report.reportedAt, freeMiB: report.freeMiB ?? null,
      largestProcess: report.largestProcess ?? null, kills: report.kills.map(kill => `${kill.name} ${kill.reason}`),
      webContent: report.webContent.map(process => ({ pid: process.pid, footprintMiB: process.footprintMiB, lifetimeMaxMiB: process.lifetimeMaxMiB })) }); }
    const candidates = rows.filter(row => !taken.has(row.index) && row.endedAt > previousAt - toleranceMs && row.endedAt <= report.reportedAt + toleranceMs);
    const chosen = kills.length ? candidates.slice(-kills.length) : [];
    const offset = kills.length - chosen.length;
    kills.forEach((kill, position) => {
      const row = position - offset >= 0 ? chosen[position - offset] : null;
      if (!row) { unmatched.push({ file: report.file, pid: kill.pid, reason: kill.reason, footprintMiB: kill.footprintMiB }); return; }
      taken.add(row.index);
      matched.push({ index: row.index, runId: row.result.runId ?? null, deviceLog: deviceLogFromKill(report, kill, now) });
    });
    previousAt = report.reportedAt;
  }
  return { matched, unmatched, context, ignored: reports.filter(report => !usable.includes(report)).map(report => ({ file: report.file, kind: report.kind, event: report.event ?? null, error: report.error ?? null })) };
}

import React, { useEffect, useRef } from 'react';

// ─────────────────────────────────────────────────────────────────────────────
// VhrJobsModal — centered overlay popup that lists every Very-HR
// local-processing job. Two sections: IN PROGRESS (with HONEST feedback —
// a real % bar only when the backend reports one, otherwise an
// indeterminate spinner; NEVER a time-extrapolated bar) and FINISHED
// (with result filename + size).
// Jobs survive reloads because the parent polls /api/very-hr-job/list.
// ─────────────────────────────────────────────────────────────────────────────
export default function VhrJobsModal({ open, onClose, jobs, onUpload, onCancel, onDelete }) {
  // Esc to close
  useEffect(() => {
    if (!open) return undefined;
    const onKey = (e) => { if (e.key === 'Escape') onClose && onClose(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  if (!open) return null;

  const all = jobs || [];
  const running = all.filter(j => j.status === 'processing');
  const finished = all.filter(j => j.status === 'done');
  const cancelled = all.filter(j => j.status === 'cancelled');

  return (
    <div
      onClick={onClose}
      style={{
        position:'fixed', inset:0, zIndex:1000,
        background:'rgba(15,23,42,0.55)',
        display:'flex', alignItems:'center', justifyContent:'center', padding:'24px',
        backdropFilter:'blur(3px)',
      }}
    >
      <div
        onClick={e => e.stopPropagation()}
        style={{
          width:'min(720px, 100%)', maxHeight:'90vh', display:'flex', flexDirection:'column',
          background:'var(--bg-primary)', borderRadius:'14px',
          border:'1px solid var(--border-dim)',
          boxShadow:'0 20px 60px rgba(15,23,42,0.35)',
          overflow:'hidden',
        }}
      >
        {/* Header */}
        <div style={{
          padding:'16px 20px', display:'flex', alignItems:'center', justifyContent:'space-between',
          background:'linear-gradient(135deg, rgba(79,70,229,0.08), rgba(124,58,237,0.06))',
          borderBottom:'1px solid var(--border-dim)',
        }}>
          <div>
            <h2 style={{margin:0,fontSize:'14px',fontFamily:'var(--font-display)',fontWeight:800,letterSpacing:'0.06em',color:'#4338ca'}}>
              📦 LOCAL-PROCESSING JOBS
            </h2>
            <p style={{margin:'4px 0 0',fontSize:'10px',fontFamily:'var(--font-mono)',color:'var(--text-dim)'}}>
              {running.length} running · {finished.length} done{cancelled.length>0?` · ${cancelled.length} cancelled`:''}
            </p>
          </div>
          <button onClick={onClose} aria-label="Close"
            style={{
              width:'30px', height:'30px', borderRadius:'8px', border:'1px solid var(--border-dim)',
              background:'var(--bg-secondary)', color:'var(--text-dim)', cursor:'pointer',
              fontSize:'14px', fontWeight:700,
            }}
          >✕</button>
        </div>

        {/* Body */}
        <div style={{padding:'16px 20px',overflow:'auto',display:'flex',flexDirection:'column',gap:'18px'}}>
          {all.length === 0 && (
            <div style={{
              padding:'30px',textAlign:'center',
              border:'1px dashed var(--border-dim)',borderRadius:'10px',
            }}>
              <p style={{margin:0,fontSize:'12px',fontFamily:'var(--font-mono)',color:'var(--text-dim)'}}>
                No local jobs yet.
              </p>
              <p style={{margin:'6px 0 0',fontSize:'11px',fontFamily:'var(--font-mono)',color:'var(--text-dim)'}}>
                Draw an ROI then click <b>Run Very HR Bathymetry (~1 m)</b>.
              </p>
            </div>
          )}

          {running.length > 0 && (
            <Section title="⏳ IN PROGRESS" count={running.length} color="#4f46e5">
              {running.map(j => (
                <JobRow key={j.id} job={j}
                  onUpload={(f)=>onUpload && onUpload(j.id, f)}
                  onCancel={()=>onCancel && onCancel(j.id)}
                  onDelete={()=>onDelete && onDelete(j.id)} />
              ))}
            </Section>
          )}

          {finished.length > 0 && (
            <Section title="✓ FINISHED" count={finished.length} color="#059669">
              {finished.map(j => (
                <JobRow key={j.id} job={j}
                  onUpload={(f)=>onUpload && onUpload(j.id, f)}
                  onCancel={()=>onCancel && onCancel(j.id)}
                  onDelete={()=>onDelete && onDelete(j.id)} />
              ))}
            </Section>
          )}

          {cancelled.length > 0 && (
            <Section title="⨯ CANCELLED" count={cancelled.length} color="#92400e">
              {cancelled.map(j => (
                <JobRow key={j.id} job={j}
                  onUpload={(f)=>onUpload && onUpload(j.id, f)}
                  onCancel={()=>onCancel && onCancel(j.id)}
                  onDelete={()=>onDelete && onDelete(j.id)} />
              ))}
            </Section>
          )}
        </div>

        {/* Footer */}
        <div style={{
          padding:'10px 20px', borderTop:'1px solid var(--border-dim)',
          background:'var(--bg-secondary)',
          display:'flex',justifyContent:'space-between',alignItems:'center',
        }}>
          <p style={{margin:0,fontSize:'9px',fontFamily:'var(--font-mono)',color:'var(--text-dim)'}}>
            Jobs are server-persisted — they survive reloads and tab closes.
          </p>
          <button onClick={onClose} style={{
            padding:'7px 14px', fontSize:'11px', fontFamily:'var(--font-display)', fontWeight:700,
            borderRadius:'6px', border:'1px solid var(--border-dim)',
            background:'var(--bg-primary)', color:'var(--text-secondary)', cursor:'pointer',
          }}>Close</button>
        </div>
      </div>
    </div>
  );
}

function Section({ title, count, color, children }) {
  return (
    <div>
      <h3 style={{
        margin:'0 0 8px',fontSize:'10px',fontFamily:'var(--font-mono)',fontWeight:800,
        letterSpacing:'0.1em',color,
      }}>{title} · {count}</h3>
      <div style={{display:'flex',flexDirection:'column',gap:'8px'}}>{children}</div>
    </div>
  );
}

function fmtHMS(s) {
  s = Math.max(0, Math.floor(s));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  return h > 0 ? `${h}h ${m}m ${sec}s` : (m > 0 ? `${m}m ${sec}s` : `${sec}s`);
}

function JobRow({ job, onUpload, onCancel, onDelete }) {
  const fileRef = useRef(null);
  const isDone = job.status === 'done';
  const isCancelled = job.status === 'cancelled';
  const isRunning = !isDone && !isCancelled;

  // HONEST progress (R1): a real % bar ONLY when the backend reports a real
  // numeric progress_pct AND does not flag the job indeterminate. Otherwise
  // we render an indeterminate spinner — we NEVER extrapolate from elapsed
  // time (that was the removed defect).
  const hasRealPct = typeof job.progress_pct === 'number' &&
                     job.indeterminate !== true &&
                     job.progress_source !== 'indeterminate';
  // Elapsed is reported honestly straight from server timestamps (it is a
  // fact, not a prediction) — the backend exposes elapsed_seconds.
  const elapsedS = typeof job.elapsed_seconds === 'number'
    ? job.elapsed_seconds
    : (isDone ? Math.max(0, (job.finished_at || 0) - (job.started_at || 0)) : 0);
  const pct = isDone ? 100 : (hasRealPct ? Math.min(100, Math.max(0, job.progress_pct)) : null);

  const statusColor = isDone ? '#065f46' : isCancelled ? '#92400e' : '#4338ca';
  const statusBg = isDone ? 'rgba(5,150,105,0.18)' : isCancelled ? 'rgba(180,83,9,0.18)' : 'rgba(79,70,229,0.18)';

  return (
    <div style={{
      padding:'12px 14px', borderRadius:'10px',
      border:'1px solid rgba(79,70,229,0.30)',
      background:'linear-gradient(135deg, rgba(79,70,229,0.04), rgba(124,58,237,0.03))',
    }}>
      <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:'8px',gap:'8px',flexWrap:'wrap'}}>
        <div style={{display:'flex',alignItems:'center',gap:'8px',minWidth:0,flex:1}}>
          <span style={{fontSize:'11px',fontFamily:'var(--font-mono)',fontWeight:800,color:'#4f46e5',whiteSpace:'nowrap'}}>
            ◆ {job.id}
          </span>
          <span style={{fontSize:'11px',fontFamily:'var(--font-mono)',color:'var(--text-primary)',overflow:'hidden',textOverflow:'ellipsis',whiteSpace:'nowrap'}}>
            {job.label || 'Very HR Bathymetry'}
          </span>
        </div>
        <span style={{
          fontSize:'9px',fontFamily:'var(--font-mono)',fontWeight:700,padding:'3px 7px',
          borderRadius:'5px',background:statusBg,color:statusColor,letterSpacing:'0.05em',
        }}>{(job.status||'').toUpperCase()}</span>
      </div>

      <div style={{display:'grid',gridTemplateColumns:'repeat(3, 1fr)',gap:'8px',fontSize:'10px',fontFamily:'var(--font-mono)',color:'var(--text-dim)',marginBottom:'10px'}}>
        <div><span>Surface </span><b style={{color:'var(--text-primary)'}}>{job.area_km2} km²</b></div>
        <div>
          <span>Stage </span>
          <b style={{color:'var(--text-primary)'}}>
            {(job.stage_index != null && job.stage_total != null)
              ? `${job.stage_index}/${job.stage_total}`
              : '—'}
          </b>
        </div>
        <div><span>Elapsed </span><b style={{color:'var(--text-primary)'}}>{fmtHMS(elapsedS)}</b></div>
      </div>

      {/* HONEST progress: a real % bar when the backend gives one; for done
          show a full bar; otherwise an indeterminate marquee — NO elapsed
          extrapolation. */}
      {(isDone || hasRealPct) ? (
        <div style={{height:'10px',width:'100%',background:'rgba(0,0,0,0.06)',borderRadius:'6px',overflow:'hidden',marginBottom:'6px'}}>
          <div style={{
            height:'100%', width:`${pct || 0}%`,
            background: isDone ? 'linear-gradient(90deg,#059669,#0d9488)' : 'linear-gradient(90deg,#7c3aed,#4f46e5)',
            transition:'width 0.5s linear',
          }} />
        </div>
      ) : isRunning ? (
        <div style={{height:'10px',width:'100%',background:'rgba(0,0,0,0.06)',borderRadius:'6px',overflow:'hidden',marginBottom:'6px',position:'relative'}}>
          <div style={{
            position:'absolute', top:0, bottom:0, width:'35%', borderRadius:'6px',
            background:'linear-gradient(90deg,rgba(124,58,237,0),#7c3aed,#4f46e5,rgba(79,70,229,0))',
            animation:'vhrIndeterminate 1.3s ease-in-out infinite',
          }} />
        </div>
      ) : (
        <div style={{height:'10px',width:'100%',background:'rgba(0,0,0,0.06)',borderRadius:'6px',marginBottom:'6px'}} />
      )}
      <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',margin:'0 0 10px',gap:'8px'}}>
        <p style={{margin:0,fontSize:'10px',fontFamily:'var(--font-mono)',color:'var(--text-dim)',flex:1,overflow:'hidden',textOverflow:'ellipsis',whiteSpace:'nowrap'}}>
          {isDone
            ? 'Result ready'
            : isCancelled
              ? (job.error ? `Error: ${job.error.slice(0,80)}` : 'Cancelled')
              : (job.stage || 'Processing…')}
        </p>
        <p style={{margin:0,fontSize:'10px',fontFamily:'var(--font-mono)',color:'var(--text-dim)',fontWeight:700}}>
          {isDone ? '100%' : hasRealPct ? `${pct.toFixed(1)}%` : '⋯'}
        </p>
      </div>

      {isDone ? (
        <div style={{padding:'8px 10px',background:'rgba(5,150,105,0.10)',borderRadius:'6px',border:'1px solid rgba(5,150,105,0.30)',display:'flex',justifyContent:'space-between',alignItems:'center',gap:'8px'}}>
          <div style={{minWidth:0}}>
            <p style={{margin:0,fontSize:'11px',fontFamily:'var(--font-mono)',fontWeight:800,color:'#065f46'}}>✓ Process Done</p>
            <p style={{margin:'3px 0 0',fontSize:'9px',fontFamily:'var(--font-mono)',color:'var(--text-dim)',overflow:'hidden',textOverflow:'ellipsis',whiteSpace:'nowrap'}}>
              {job.result_filename} · {Math.round((job.result_size_bytes||0)/1024)} KB
            </p>
          </div>
          <button onClick={onDelete} style={{
            padding:'6px 10px',fontSize:'10px',fontFamily:'var(--font-mono)',fontWeight:700,borderRadius:'5px',
            border:'1px solid var(--border-dim)',background:'var(--bg-primary)',color:'var(--text-dim)',cursor:'pointer',whiteSpace:'nowrap',
          }}>Remove</button>
        </div>
      ) : (
        <>
          <input ref={fileRef} type="file" style={{display:'none'}}
            onChange={e=>{
              const f = e.target.files && e.target.files[0];
              if (f) onUpload(f);
              if (e.target) e.target.value = '';
            }} />
          <div style={{display:'flex',gap:'6px'}}>
            <button onClick={()=>fileRef.current && fileRef.current.click()}
              disabled={isCancelled}
              style={{
                flex:1, padding:'8px', fontSize:'10px', fontFamily:'var(--font-display)',
                fontWeight:700, borderRadius:'5px', border:'1px solid #4f46e5',
                background: isCancelled ? 'var(--bg-secondary)' : '#4f46e5',
                color: isCancelled ? 'var(--text-dim)' : '#fff',
                cursor: isCancelled ? 'not-allowed' : 'pointer', letterSpacing:'0.4px',
              }}>📤 Upload result</button>
            <button onClick={isCancelled ? onDelete : onCancel} style={{
              padding:'8px 12px', fontSize:'10px', fontFamily:'var(--font-mono)',
              fontWeight:700, borderRadius:'5px', border:'1px solid var(--border-dim)',
              background:'var(--bg-primary)', color:'var(--text-dim)', cursor:'pointer',
            }}>{isCancelled ? 'Remove' : 'Cancel'}</button>
          </div>
        </>
      )}
    </div>
  );
}

"""Label-free SIFE44 and its Pure36 subset, with explicit typed edge inputs.

The numeric formulas reproduce the reviewed v9 definition on valid finite,
nonnegative count/sum records. Invalid incident weights are rejected rather
than silently repaired. No source14 values or labels enter SIFE computation.
"""
import numpy as np

FEATURE_VERSION = 'sife44_distribution_v9_formulas_v1'
CHANNELS = (('in','call'), ('out','call'), ('in','trans'), ('out','trans'))
DISTRIBUTION_STATS = ('entropy','entropy_norm','hhi','jsd_freq_val','max_share',
                      'std_log_val','mean_tx_val')
SIFE44_NAMES = tuple(
    [f'sife_{rel}_{direction}_{stat}' for direction,rel in CHANNELS for stat in DISTRIBUTION_STATS]
    + [f'sife_{rel}_{stat}' for rel in ('call','trans')
       for stat in ('funnel_entropy','funnel_hhi','asym_deg','asym_flow')]
    + [f'sife_{rel}_{direction}_{stat}' for direction,rel in CHANNELS
       for stat in ('ca_edge_ratio','ca_count_ratio')])
PURE36_INDICES = tuple(ch*7+j for ch in range(4) for j in range(5))+tuple(range(28,44))
INTENSITY8_INDICES = tuple(ch*7+j for ch in range(4) for j in (5,6))
FEATURE_GROUPS = {
    'distribution20':tuple(ch*7+j for ch in range(4) for j in range(5)),
    'flow8':tuple(range(28,36)),
    'counterparty8':tuple(range(36,44)),
    'intensity8':INTENSITY8_INDICES,
}
ARMS = ('X14_SIFE44','without_distribution20','without_flow8','without_counterparty8','X14_Pure36')
ARM_REMOVED_GROUP = {'X14_SIFE44':None,'without_distribution20':'distribution20',
    'without_flow8':'flow8','without_counterparty8':'counterparty8','X14_Pure36':'intensity8'}


def selected_sife_indices(arm):
    """Ordered kept SIFE columns; compatibility X14/Pure36 are not active study arms."""
    if arm=='X14':return ()
    if arm=='Pure36':return PURE36_INDICES
    if arm not in ARM_REMOVED_GROUP:raise ValueError('Unknown feature arm')
    removed=FEATURE_GROUPS.get(ARM_REMOVED_GROUP[arm],())
    return tuple(i for i in range(44) if i not in removed)


def safe_ratio(numerator, denominator):
    """Return zero for a zero denominator, retaining signed contrast numerators."""
    return np.divide(numerator,denominator,out=np.zeros_like(numerator,dtype=np.float64),
                     where=denominator>0)


def summarize_channel(n, record_factory):
    """Two passes over (local_id,count,sum,partner_is_CA) for a direction/relation.

    For a nonempty zero-mass channel, use a uniform distribution on its records.
    For an absent channel, all returned statistics are zero. JSD uses natural logs.
    """
    rows,counts,sums,maximum,logs,ca_rows,ca_counts=np.zeros((7,n),dtype=np.float64)
    for idx,c_raw,s_raw,partner_ca in record_factory():
        c=np.asarray(c_raw,dtype=np.float64)
        s=np.asarray(s_raw,dtype=np.float64)
        if not (np.isfinite(c).all() and np.isfinite(s).all() and (c>=0).all() and (s>=0).all()):
            raise ValueError('Incident count/sum weights must be finite and nonnegative')
        rows+=np.bincount(idx,minlength=n)
        counts+=np.bincount(idx,weights=c,minlength=n)
        sums+=np.bincount(idx,weights=s,minlength=n)
        logs+=np.bincount(idx,weights=np.log1p(s),minlength=n)
        np.maximum.at(maximum,idx,s)
        if partner_ca:
            ca_rows+=np.bincount(idx,minlength=n)
            ca_counts+=np.bincount(idx,weights=c,minlength=n)
    entropy,hhi,jsd,variance=np.zeros((4,n),dtype=np.float64)
    mean_log=safe_ratio(logs,rows)
    for idx,c_raw,s_raw,_ in record_factory():
        c=np.asarray(c_raw,dtype=np.float64)
        s=np.asarray(s_raw,dtype=np.float64)
        pc=safe_ratio(c,counts[idx])
        ps=safe_ratio(s,sums[idx])
        pc=np.where(counts[idx]>0,pc,1/rows[idx])
        ps=np.where(sums[idx]>0,ps,1/rows[idx])
        midpoint=(pc+ps)/2
        terms=np.zeros_like(ps)
        positive=ps>0
        terms[positive]=-ps[positive]*np.log(ps[positive])
        entropy+=np.bincount(idx,weights=terms,minlength=n)
        hhi+=np.bincount(idx,weights=ps**2,minlength=n)
        terms=np.zeros_like(ps)
        for probability in (pc,ps):
            positive=probability>0
            terms[positive]+=.5*probability[positive]*np.log(probability[positive]/midpoint[positive])
        jsd+=np.bincount(idx,weights=terms,minlength=n)
        variance+=np.bincount(idx,weights=(np.log1p(s)-mean_log[idx])**2,minlength=n)
    normalized=safe_ratio(entropy,np.log(np.maximum(rows,2)))*(rows>1)
    dominance=np.where(sums>0,safe_ratio(maximum,sums),safe_ratio(np.ones(n),rows))
    distribution=np.column_stack((entropy,normalized,hhi,np.clip(jsd,0,np.log(2)),dominance,
                    np.sqrt(safe_ratio(variance,rows)),np.log1p(safe_ratio(sums,counts))))
    metadata=dict(rows=rows,counts=counts,sums=sums,entropy=entropy,hhi=hhi,
                  ca_rows=safe_ratio(ca_rows,rows),ca_counts=safe_ratio(ca_counts,counts))
    return distribution,metadata


def extract_sife44(node_counts, edges, node_type, ids, chunk_size=250000):
    """Compute all incident records for sorted IDs, independent of labels/source14.

    edges maps (source_type,call|trans,destination_type) to arrays edge_index[2,E]
    and edge_attr[E,2] in the same order; the attribute columns are count and sum.
    Already aggregated typed counterpart pairs must retain their original identity.
    The prepared-data stage separately verifies pair uniqueness and raw alignment.
    """
    if node_type not in ('EOA','CA') or node_type not in node_counts:
        raise ValueError('Known EOA/CA target node type required')
    ids=np.asarray(ids)
    if ids.ndim!=1 or not np.issubdtype(ids.dtype,np.integer) or not len(ids):
        raise ValueError('Nonempty integer node ID vector required')
    ids=ids.astype(np.int64,copy=False)
    if np.any(np.diff(ids)<=0) or ids[0]<0 or ids[-1]>=node_counts[node_type]:
        raise ValueError('Node IDs must be sorted, unique and in range')
    if not isinstance(chunk_size,int) or chunk_size<1:
        raise ValueError('Positive integer chunk size required')
    for (src,rel,dst),record in edges.items():
        ix=record['edge_index']
        attr=record.get('edge_attr')
        if src not in node_counts or dst not in node_counts or rel not in ('call','trans'):
            raise ValueError('Unsupported typed relation')
        if ix.ndim!=2 or ix.shape[0]!=2 or not np.issubdtype(ix.dtype,np.integer):
            raise ValueError('edge_index must have integer shape[2,E]')
        if attr is None or attr.shape!=(ix.shape[1],2):
            raise ValueError('Aligned count/sum edge_attr[E,2] is required')
    lookup=np.full(node_counts[node_type],-1,dtype=np.int32)
    lookup[ids]=np.arange(len(ids),dtype=np.int32)
    values=np.zeros((len(ids),44),dtype=np.float64)
    metadata={}
    for channel,(direction,relation) in enumerate(CHANNELS):
        def factory():
            for (src,rel,dst),record in edges.items():
                if rel!=relation or (src if direction=='out' else dst)!=node_type:
                    continue
                axis=0 if direction=='out' else 1
                partner=dst if direction=='out' else src
                for start in range(0,record['edge_index'].shape[1],chunk_size):
                    endpoints=record['edge_index'][axis,start:start+chunk_size]
                    if endpoints.size and (endpoints.min()<0 or endpoints.max()>=len(lookup)):
                        raise ValueError('Target endpoint out of range')
                    local=lookup[endpoints]
                    selected=local>=0
                    attr=record['edge_attr'][start:start+chunk_size][selected]
                    yield local[selected],attr[:,0],attr[:,1],partner=='CA'
        dist,meta=summarize_channel(len(ids),factory)
        values[:,channel*7:(channel+1)*7]=dist
        metadata[(direction,relation)]=meta
        values[:,36+channel*2]=meta['ca_rows']
        values[:,37+channel*2]=meta['ca_counts']
    for j,relation in enumerate(('call','trans')):
        incoming=metadata[('in',relation)]
        outgoing=metadata[('out',relation)]
        values[:,28+j*4:32+j*4]=np.column_stack((
            incoming['entropy']-outgoing['entropy'],outgoing['hhi']-incoming['hhi'],
            safe_ratio(incoming['rows']-outgoing['rows'],incoming['rows']+outgoing['rows']),
            safe_ratio(incoming['sums']-outgoing['sums'],incoming['sums']+outgoing['sums'])))
    activity=np.column_stack((np.log1p(sum(m['counts'] for m in metadata.values())),
                              np.log1p(sum(m['sums'] for m in metadata.values()))))
    if not np.isfinite(values).all() or not np.isfinite(activity).all():
        raise ValueError('Nonfinite feature output')
    return dict(ids=ids.copy(),sife44=values,activity=activity)


def select_inputs(source14, sife44, arm, model=None):
    """Select explicitly named blocks; no historical cache column offsets."""
    selected=selected_sife_indices(arm)
    if arm=='Pure36' and model=='metaifd':
        raise ValueError('Meta-IFD generator requires source14; Pure36-only is unsupported')
    sife44=np.asarray(sife44)
    if sife44.ndim!=2 or sife44.shape[1]!=44 or not np.isfinite(sife44).all():
        raise ValueError('Finite SIFE44 matrix required')
    if arm=='Pure36':
        return sife44[:,PURE36_INDICES].copy()
    source14=np.asarray(source14)
    if source14.shape!=(len(sife44),14) or not np.isfinite(source14).all():
        raise ValueError('Aligned finite source14 required')
    if arm=='X14':
        return source14.copy()
    extra=sife44[:,selected]
    return np.column_stack((source14,extra))

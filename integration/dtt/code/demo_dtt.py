import os, uuid, json
os.environ["CQE_GOV_MOD"] = "cqe_core.dual_governance"
from cqe_core.dtt_orchestrator import default as dtt

pkt = {
    'id': f'idea:{uuid.uuid4()}',
    'type': 'retrieval_strategy',
    'content': {'algorithm': 'hybrid_semantic_lattice'},
    'context': {'sample_query': 'nearest lattice rounding'},
    'expected_outputs': {'precision_gain': '>=0.05'},
    'metadata': {'origin': 'manual-demo'}
}

dtt.submit(pkt)
print("Submitted idea packet", pkt['id'])

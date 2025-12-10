#!/usr/bin/env python3
"""
CQE Merit-Based Knowledge Valuation System
==========================================
Realistic tiered value assessment using civilization simulation.
Most work gets micro-coins. Revolutionary work gets major rewards.
1 merit = 1 local currency unit (stable by design).
"""

import numpy as np
import hashlib
import json
from typing import Dict, List, Tuple, Optional
from enum import Enum
from dataclasses import dataclass

class KnowledgeTier(Enum):
    """
    Realistic knowledge value tiers based on civilizational impact
    """
    NOISE = 0           # Spam, duplicates, errors: 0 merit
    COLLOQUIAL = 1      # Everyday discussion: 0.001 merit (micro-coins)
    USEFUL = 2          # Practical tips: 0.01 merit  
    TECHNICAL = 3       # Professional work: 0.1 merit
    SCHOLARLY = 4       # Peer-reviewed research: 1 merit
    INNOVATIVE = 5      # Novel contributions: 10 merits
    BREAKTHROUGH = 6    # Paradigm shifts: 100 merits
    REVOLUTIONARY = 7   # Civilization-changing: 1000+ merits

@dataclass
class CitationRecord:
    """Track academic citations for identity and value"""
    author_id: str
    paper_id: str
    citation_count: int
    h_index: float
    field_impact: float
    verified: bool

class CivilizationSimulator:
    """
    Simulate multiple civilizations to assess knowledge value.
    This is the key innovation - value determined by utility across civs.
    """
    
    def __init__(self):
        # Simulate different civilization types
        self.civs = {
            'agricultural': {'tech_level': 1, 'priorities': ['food', 'water', 'shelter']},
            'industrial': {'tech_level': 3, 'priorities': ['energy', 'manufacturing', 'transport']},
            'information': {'tech_level': 5, 'priorities': ['data', 'networks', 'algorithms']},
            'space': {'tech_level': 7, 'priorities': ['physics', 'engineering', 'biology']},
            'post-scarcity': {'tech_level': 9, 'priorities': ['philosophy', 'art', 'consciousness']}
        }
        
    def evaluate_knowledge_value(self, content: Dict) -> Tuple[KnowledgeTier, float]:
        """
        Evaluate knowledge across multiple civilizations.
        Value emerges from cross-civilizational utility.
        """
        scores = []
        
        for civ_name, civ in self.civs.items():
            # How useful is this knowledge to this civilization?
            relevance = self._calculate_relevance(content, civ['priorities'])
            advancement = self._calculate_advancement_potential(content, civ['tech_level'])
            
            # Weight by civilization's development level
            civ_score = (relevance * advancement) * (civ['tech_level'] / 10)
            scores.append(civ_score)
        
        # Average utility across all civilizations
        avg_score = np.mean(scores)
        std_score = np.std(scores)
        
        # Determine tier based on score distribution
        if avg_score < 0.001:
            return KnowledgeTier.NOISE, 0.0
        elif avg_score < 0.01:
            return KnowledgeTier.COLLOQUIAL, 0.001
        elif avg_score < 0.1:
            return KnowledgeTier.USEFUL, 0.01
        elif avg_score < 1.0:
            return KnowledgeTier.TECHNICAL, 0.1
        elif avg_score < 10:
            return KnowledgeTier.SCHOLARLY, 1.0
        elif avg_score < 100:
            return KnowledgeTier.INNOVATIVE, 10.0
        elif std_score > 50:  # High variance = revolutionary
            return KnowledgeTier.REVOLUTIONARY, 1000.0
        else:
            return KnowledgeTier.BREAKTHROUGH, 100.0
    
    def _calculate_relevance(self, content: Dict, priorities: List[str]) -> float:
        """How relevant is this content to civilization's priorities?"""
        # Simplified: check keyword overlap
        content_keywords = content.get('keywords', [])
        overlap = len(set(content_keywords) & set(priorities))
        return overlap / max(len(priorities), 1)
    
    def _calculate_advancement_potential(self, content: Dict, tech_level: int) -> float:
        """Can this knowledge advance the civilization?"""
        content_level = content.get('complexity', 1)
        
        # Too advanced = not useful yet
        if content_level > tech_level + 2:
            return 0.1
        # Too primitive = already known
        elif content_level < tech_level - 2:
            return 0.1
        # Just right = maximum value
        else:
            return 1.0 - abs(content_level - tech_level) * 0.2

class MeritCoinSystem:
    """
    Realistic merit-based coin distribution.
    1 merit = 1 local currency unit (stable).
    """
    
    def __init__(self):
        self.civ_sim = CivilizationSimulator()
        self.total_supply = 0
        self.distribution = {tier: 0 for tier in KnowledgeTier}
        self.citation_ledger = {}
        self.identity_proofs = {}
        
    def verify_identity(self, user_id: str, credentials: Dict) -> bool:
        """
        Verify identity through academic citations or contributions.
        """
        # Check for academic citations
        if 'orcid' in credentials:
            citations = self._fetch_citations(credentials['orcid'])
            if citations:
                self.citation_ledger[user_id] = citations
                return True
        
        # Check for verified uploads
        if 'previous_uploads' in credentials:
            verified_count = sum(1 for u in credentials['previous_uploads'] 
                               if self._verify_upload(u))
            if verified_count > 0:
                self.identity_proofs[user_id] = verified_count
                return True
        
        # Even anecdotal work counts (but gets micro-rewards)
        if 'anecdotal_proof' in credentials:
            self.identity_proofs[user_id] = 0.001  # Minimal but real
            return True
        
        return False
    
    def evaluate_contribution(self, content: Dict, user_id: str) -> Dict:
        """
        Evaluate contribution and mint appropriate coins.
        Most will be micro-coins. Very few will be substantial.
        """
        # Get civilization simulation assessment
        tier, base_value = self.civ_sim.evaluate_knowledge_value(content)
        
        # Adjust for citations and credibility
        credibility_multiplier = 1.0
        if user_id in self.citation_ledger:
            citations = self.citation_ledger[user_id]
            # High h-index increases value (but not linearly)
            credibility_multiplier = 1.0 + np.log10(citations.h_index + 1)
        
        # Calculate final merit value
        merit_value = base_value * credibility_multiplier
        
        # Enforce stable value: 1 merit = 1 local currency
        # This means controlling supply firmly
        if self._would_cause_inflation(merit_value):
            merit_value = self._adjust_for_stability(merit_value)
        
        # Update distribution
        self.distribution[tier] += 1
        self.total_supply += merit_value
        
        # Escrow for citations
        escrow_amount = 0
        if self._has_citations_on_node_work(user_id):
            escrow_amount = merit_value * 0.1  # 10% held for citation rewards
        
        return {
            'tier': tier.name,
            'merit_value': merit_value,
            'escrow': escrow_amount,
            'distributed': merit_value - escrow_amount,
            'total_supply': self.total_supply,
            'stability_ratio': self._calculate_stability_ratio()
        }
    
    def _would_cause_inflation(self, new_merits: float) -> bool:
        """Check if minting would destabilize the 1:1 peg"""
        # Complex economic model would go here
        # For now: simple supply growth limit
        growth_rate = new_merits / (self.total_supply + 1)
        return growth_rate > 0.001  # Max 0.1% growth per mint
    
    def _adjust_for_stability(self, merit_value: float) -> float:
        """Adjust merit value to maintain 1:1 currency peg"""
        max_mint = self.total_supply * 0.001  # 0.1% max
        return min(merit_value, max_mint)
    
    def _calculate_stability_ratio(self) -> float:
        """Monitor the 1 merit = 1 currency stability"""
        # Would connect to exchange rates in production
        return 1.0  # Placeholder for stable peg
    
    def _fetch_citations(self, orcid: str) -> Optional[CitationRecord]:
        """Fetch real citation data (would connect to academic databases)"""
        # Simulated for demo
        return CitationRecord(
            author_id=orcid,
            paper_id="mock",
            citation_count=np.random.randint(0, 100),
            h_index=np.random.randint(0, 50),
            field_impact=np.random.random(),
            verified=True
        )
    
    def _verify_upload(self, upload_hash: str) -> bool:
        """Verify previous upload is genuine"""
        # Would check blockchain history
        return len(upload_hash) == 64  # Simple hash check
    
    def _has_citations_on_node_work(self, user_id: str) -> bool:
        """Check if user's work is cited in any node"""
        # Would query the network
        return user_id in self.citation_ledger

def demonstrate_realistic_distribution():
    """
    Show realistic knowledge value distribution.
    Most content is mundane, very little is revolutionary.
    """
    print("="*60)
    print("REALISTIC MERIT-BASED KNOWLEDGE VALUATION")
    print("="*60)
    
    system = MeritCoinSystem()
    
    # Simulate 10,000 contributions (realistic distribution)
    contributions = []
    
    # 90% is colloquial/everyday discussion
    for _ in range(9000):
        content = {
            'type': 'discussion',
            'keywords': ['daily', 'chat', 'opinion'],
            'complexity': 1
        }
        result = system.evaluate_contribution(content, f"user_{_}")
        contributions.append(result)
    
    # 8% is useful/practical
    for _ in range(800):
        content = {
            'type': 'tutorial',
            'keywords': ['howto', 'guide', 'practical'],
            'complexity': 2
        }
        result = system.evaluate_contribution(content, f"pro_{_}")
        contributions.append(result)
    
    # 1.5% is technical/professional
    for _ in range(150):
        content = {
            'type': 'technical',
            'keywords': ['algorithm', 'design', 'engineering'],
            'complexity': 4
        }
        result = system.evaluate_contribution(content, f"tech_{_}")
        contributions.append(result)
    
    # 0.4% is scholarly
    for _ in range(40):
        content = {
            'type': 'research',
            'keywords': ['hypothesis', 'methodology', 'analysis'],
            'complexity': 6
        }
        result = system.evaluate_contribution(content, f"scholar_{_}")
        contributions.append(result)
    
    # 0.09% is innovative
    for _ in range(9):
        content = {
            'type': 'innovation',
            'keywords': ['novel', 'breakthrough', 'discovery'],
            'complexity': 8
        }
        result = system.evaluate_contribution(content, f"innovator_{_}")
        contributions.append(result)
    
    # 0.01% is revolutionary (1 in 10,000)
    content = {
        'type': 'paradigm_shift',
        'keywords': ['physics', 'consciousness', 'mathematics', 'energy'],
        'complexity': 10
    }
    result = system.evaluate_contribution(content, "genius_0")
    contributions.append(result)
    
    # Show distribution
    print("\n1. CONTRIBUTION DISTRIBUTION")
    print("-"*40)
    
    for tier in KnowledgeTier:
        count = system.distribution[tier]
        percentage = (count / len(contributions)) * 100 if contributions else 0
        print(f"{tier.name:14} : {count:6} ({percentage:5.2f}%)")
    
    # Show economic metrics
    print("\n2. ECONOMIC METRICS")
    print("-"*40)
    
    total_merits = sum(c['merit_value'] for c in contributions)
    avg_merit = total_merits / len(contributions)
    median_merit = np.median([c['merit_value'] for c in contributions])
    
    print(f"Total supply: {total_merits:.2f} merits")
    print(f"Average value: {avg_merit:.6f} merits")
    print(f"Median value: {median_merit:.6f} merits")
    print(f"Stability ratio: {system._calculate_stability_ratio():.2f} (1.0 = perfect)")
    
    # Show wealth distribution (should follow Pareto principle)
    print("\n3. WEALTH DISTRIBUTION")
    print("-"*40)
    
    sorted_contributions = sorted(contributions, key=lambda x: x['merit_value'], reverse=True)
    top_1_percent = int(len(contributions) * 0.01)
    top_10_percent = int(len(contributions) * 0.10)
    
    top_1_wealth = sum(c['merit_value'] for c in sorted_contributions[:top_1_percent])
    top_10_wealth = sum(c['merit_value'] for c in sorted_contributions[:top_10_percent])
    
    print(f"Top 1% owns: {(top_1_wealth/total_merits)*100:.1f}% of wealth")
    print(f"Top 10% owns: {(top_10_wealth/total_merits)*100:.1f}% of wealth")
    print(f"Bottom 90% owns: {((total_merits-top_10_wealth)/total_merits)*100:.1f}% of wealth")
    
    # Show citation escrow system
    print("\n4. CITATION ESCROW SYSTEM")
    print("-"*40)
    
    total_escrow = sum(c['escrow'] for c in contributions)
    print(f"Total in escrow: {total_escrow:.2f} merits")
    print(f"For cited authors: Reserved for verification")
    print(f"Anecdotal work: Micro-rewards but counted")
    
    print("\n5. KEY INSIGHTS")
    print("-"*40)
    
    insights = [
        f"• Most information ({9000/100:.0f}%) is only colloquially useful",
        f"• Revolutionary insights are extremely rare (0.01%)",
        f"• 1 merit = 1 {system._get_local_currency()} (stable by design)",
        f"• Citations create escrow for verified contributors",
        f"• Even anecdotal work gets micro-rewards",
        f"• Value determined by multi-civilization simulation",
        f"• Supply firmly controlled to prevent inflation"
    ]
    
    for insight in insights:
        print(insight)
    
    return system

def main():
    system = demonstrate_realistic_distribution()
    
    print("\n" + "="*60)
    print("IDENTITY VERIFICATION EXAMPLE")
    print("="*60)
    
    # Example: Academic with citations
    academic_id = "researcher_001"
    academic_creds = {
        'orcid': '0000-0001-2345-6789',
        'previous_uploads': ['hash1', 'hash2']
    }
    
    if system.verify_identity(academic_id, academic_creds):
        print(f"✓ {academic_id} verified via ORCID citations")
        
        # Submit scholarly work
        scholarly_work = {
            'type': 'research',
            'keywords': ['quantum', 'computing', 'algorithm'],
            'complexity': 7
        }
        
        result = system.evaluate_contribution(scholarly_work, academic_id)
        print(f"  Minted: {result['merit_value']:.3f} merits")
        print(f"  Escrowed: {result['escrow']:.3f} merits")
    
    # Example: Anecdotal contributor  
    anecdotal_id = "storyteller_042"
    anecdotal_creds = {
        'anecdotal_proof': 'personal experience shared'
    }
    
    if system.verify_identity(anecdotal_id, anecdotal_creds):
        print(f"✓ {anecdotal_id} verified via anecdotal proof")
        
        # Submit colloquial work
        story = {
            'type': 'anecdote',
            'keywords': ['experience', 'observation'],
            'complexity': 1
        }
        
        result = system.evaluate_contribution(story, anecdotal_id)
        print(f"  Minted: {result['merit_value']:.6f} merits (micro-coin)")

if __name__ == "__main__":
    main()

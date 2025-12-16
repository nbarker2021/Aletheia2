# Canonical Rules — Side-by-Side Citations (Top 30)
_Version: 120825_042013_

## 1. raise TypeError("Permutation hash must be int")

**Source:** System_AGRM_CMPLX_DeepReview_Part8_version_120825_022806.md:L3817 [UNK/code]
```text
L3816:     else:
L3817:         raise TypeError("Permutation must be int or tuple")
L3818: 
```

**Source:** System_AGRM_CMPLX_DeepReview_Part10_version_120825_022806.md:L19011 [UNK/code]
```text
L19010:     else:
L19011:         raise TypeError("Permutation must be int or tuple")
L19012: 
```

**Source:** System_AGRM_CMPLX_DeepReview_Part13_version_120825_022806.md:L15929 [UNK/code]
```text
L15928:     else:
L15929:         raise TypeError("Permutation must be int or tuple")
L15930: 
```

## 2. raise ValueError("ops must be a list")

**Source:** System_AGRM_CMPLX_DeepReview_Part30_version_120825_022806.md:L1920 [UNK/code]
```text
L1919:     if "|" not in k:
L1920:         raise ValueError("Edge key must be 'pre|suf'")
L1921:     pre, suf = k.split("|", 1)
```

**Source:** System_AGRM_CMPLX_DeepReview_Part31_version_120825_022806.md:L10431 [UNK/code]
```text
L10430:         if not isinstance(payload.get("tokens"), list):
L10431:             raise ValueError("tokens must be a list")
L10432: 
```

**Source:** System_AGRM_CMPLX_DeepReview_Part32_version_120825_022806.md:L139 [UNK/code]
```text
L138:     if cfg.floors_per_building < 1:
L139:         raise ValueError("floors_per_building must be >= 1")
L140:     if cfg.rooms_per_floor < 2:
```

## 3. Value should be (actual_value, metadata_dict) for AGRM integration.

**Source:** System_AGRM_CMPLX_DeepReview_Part5_version_120825_022806.md:L5496 [UNK/code]
```text
L5495:         Handles routing, velocity region, core, collisions, and conflict structures.
L5496:         Value should be (actual_value, metadata_dict) for AGRM integration.
L5497:         """
```

**Source:** System_AGRM_CMPLX_DeepReview_Part6_version_120825_022806.md:L2224 [UNK/code]
```text
L2223:         Handles routing, velocity region, core, collisions, and conflict structures.
L2224:         Value should be (actual_value, metadata_dict) for AGRM integration.
L2225:         """
```

**Source:** System_AGRM_CMPLX_DeepReview_Part8_version_120825_022806.md:L13508 [UNK/code]
```text
L13507:         Handles routing, velocity region, core, collisions, and conflict structures.
L13508:         Value should be (actual_value, metadata_dict) for AGRM integration.
L13509:         """
```

## 4. if len(kmer1) == k - 1 and len(kmer2) == k-1: #Should always be true

**Source:** System_AGRM_CMPLX_DeepReview_Part8_version_120825_022806.md:L5476 [UNK/code]
```text
L5475:               kmer2 = perm_str[i + 1:i + k]
L5476:               if len(kmer1) == k - 1 and len(kmer2) == k-1: #Should always be true
L5477:                 graph.add_edge(kmer1, kmer2, weight=calculate_overlap(kmer1, kmer2))
```

**Source:** System_AGRM_CMPLX_DeepReview_Part10_version_120825_022806.md:L20670 [UNK/code]
```text
L20669:               kmer2 = perm_str[i + 1:i + k]
L20670:               if len(kmer1) == k - 1 and len(kmer2) == k-1: #Should always be true
L20671:                 graph.add_edge(kmer1, kmer2, weight=calculate_overlap(kmer1, kmer2))
```

**Source:** System_AGRM_CMPLX_DeepReview_Part13_version_120825_022806.md:L10938 [UNK/code]
```text
L10937:               kmer2 = perm_str[i + 1:i + k]
L10938:               if len(kmer1) == k - 1 and len(kmer2) == k-1: #Should always be true
L10939:                 graph.add_edge(kmer1, kmer2, weight=calculate_overlap(kmer1, kmer2))
```

## 5. if perm_hash not in eput: #Should now always be the case

**Source:** System_AGRM_CMPLX_DeepReview_Part8_version_120825_022806.md:L4863 [UNK/code]
```text
L4862:             perm_hash = missing_perm_hash
L4863:             if perm_hash not in eput: #Should now always be the case.
L4864:                eput[perm_hash] = PermutationData(missing_perm, in_sample = False, creation_method="completion")
```

**Source:** System_AGRM_CMPLX_DeepReview_Part10_version_120825_022806.md:L20057 [UNK/code]
```text
L20056:             perm_hash = missing_perm_hash
L20057:             if perm_hash not in eput: #Should now always be the case.
L20058:                eput[perm_hash] = PermutationData(missing_perm, in_sample = False, creation_method="completion")
```

**Source:** System_AGRM_CMPLX_DeepReview_Part13_version_120825_022806.md:L11444 [UNK/code]
```text
L11443:             perm_hash = missing_perm_hash
L11444:             if perm_hash not in eput: #Should now always be the case.
L11445:                eput[perm_hash] = PermutationData(missing_perm, in_sample = False, creation_method="completion")
```

## 6. n (int): The value of n (should be 7).

**Source:** System_AGRM_CMPLX_DeepReview_Part8_version_120825_022806.md:L5857 [UNK/code]
```text
L5856:         layout_memory (LayoutMemory): The LayoutMemory object.
L5857:         n (int): The value of n (should be 7).
L5858:         golden_ratio_points (list): List of golden ratio points.
```

**Source:** System_AGRM_CMPLX_DeepReview_Part13_version_120825_022806.md:L12841 [UNK/code]
```text
L12840:         layout_memory (LayoutMemory): The LayoutMemory object.
L12841:         n (int): The value of n (should be 7).
L12842:         golden_ratio_points (list): List of golden ratio points.
```

**Source:** System_AGRM_CMPLX_DeepReview_Part15_version_120825_022806.md:L18151 [UNK/code]
```text
L18150:         layout_memory (LayoutMemory): The LayoutMemory object.
L18151:         n (int): The value of n (should be 7).
L18152:         golden_ratio_points (list): List of golden ratio points.
```

## 7. if not isinstance(payload.get("ops"), list): raise ValueError("ops must be a list")

**Source:** System_AGRM_CMPLX_DeepReview_Part9_version_120825_022806.md:L13560 [UNK/code]
```text
L13559:     def validate(self, payload): 
L13560:         if not isinstance(payload.get("tokens"), list): raise ValueError("tokens must be a list")
L13561:     def adopt(self, payload): return {"replay_tokens": payload.get("tokens", []), "vocab": payload.get("vocab", {})}
```

**Source:** System_AGRM_CMPLX_DeepReview_Part11_version_120825_022806.md:L15878 [UNK/code]
```text
L15877:     def validate(self, payload): 
L15878:         if not isinstance(payload.get("tokens"), list): raise ValueError("tokens must be a list")
L15879:     def adopt(self, payload): return {"replay_tokens": payload.get("tokens", []), "vocab": payload.get("vocab", {})}
```

**Source:** System_AGRM_CMPLX_DeepReview_Part18_version_120825_022806.md:L10609 [UNK/code]
```text
L10608:     def validate(self, payload): 
L10609:         if not isinstance(payload.get("tokens"), list): raise ValueError("tokens must be a list")
L10610:     def adopt(self, payload): return {"replay_tokens": payload.get("tokens", []), "vocab": payload.get("vocab", {})}
```

## 8. self.in_sample: bool = in_sample  # Will always be False

**Source:** System_AGRM_CMPLX_DeepReview_Part8_version_120825_022806.md:L7131 [UNK/code]
```text
L7130:         self.permutation: tuple = permutation
L7131:         self.in_sample: bool = in_sample  # Will always be False
L7132:         self.used_count: int = 0
```

**Source:** System_AGRM_CMPLX_DeepReview_Part21_version_120825_022806.md:L9736 [UNK/code]
```text
L9735:         self.permutation: tuple = permutation
L9736:         self.in_sample: bool = in_sample  # Will always be False
L9737:         self.used_count: int = 0
```

**Source:** System_AGRM_CMPLX_DeepReview_Part29_version_120825_022806.md:L18661 [UNK/code]
```text
L18660:         self.permutation: tuple = permutation
L18661:         self.in_sample: bool = in_sample  # Will always be False
L18662:         self.used_count: int = 0
```

## 9. if perm_hash not in eput:  # Should always be true here

**Source:** System_AGRM_CMPLX_DeepReview_Part8_version_120825_022806.md:L7456 [UNK/code]
```text
L7455:             perm_string = "".join(str(x) for x in best_candidate_perm)
L7456:             if perm_hash not in eput:  # Should always be true here
L7457:                 eput[perm_hash] = PermutationData(best_candidate_perm, in_sample=False, creation_method="dynamic_generation")
```

**Source:** System_AGRM_CMPLX_DeepReview_Part21_version_120825_022806.md:L10061 [UNK/code]
```text
L10060:             perm_string = "".join(str(x) for x in best_candidate_perm)
L10061:             if perm_hash not in eput:  # Should always be true here
L10062:                 eput[perm_hash] = PermutationData(best_candidate_perm, in_sample=False, creation_method="dynamic_generation")
```

**Source:** System_AGRM_CMPLX_DeepReview_Part29_version_120825_022806.md:L18986 [UNK/code]
```text
L18985:             perm_string = "".join(str(x) for x in best_candidate_perm)
L18986:             if perm_hash not in eput:  # Should always be true here
L18987:                 eput[perm_hash] = PermutationData(best_candidate_perm, in_sample=False, creation_method="dynamic_generation")
```

## 10. - L452: **AGRM** …         Value should be (actual_value, metadata_dict) for AGRM integration.

**Source:** System_AGRM_CMPLX_DeepReview_Part5_version_120825_022806.md:L5018 [UNK/code]
```text
L5017:   - L408: **MDHG** …         if not self.buildings: raise ValueError("MDHGHashTable has no buildings initialized.")
L5018:   - L452: **AGRM** …         Value should be (actual_value, metadata_dict) for AGRM integration.
L5019:   - L461: **MDHG** …             if not self.buildings: raise RuntimeError("MDHG Hash Table has no buildings.")
```

**Source:** System_AGRM_CMPLX_DeepReview_Part6_version_120825_022806.md:L1746 [UNK/code]
```text
L1745:   - L408: **MDHG** …         if not self.buildings: raise ValueError("MDHGHashTable has no buildings initialized.")
L1746:   - L452: **AGRM** …         Value should be (actual_value, metadata_dict) for AGRM integration.
L1747:   - L461: **MDHG** …             if not self.buildings: raise RuntimeError("MDHG Hash Table has no buildings.")
```

**Source:** System_AGRM_CMPLX_DeepReview_Part13_version_120825_022806.md:L16480 [UNK/code]
```text
L16479:   - L408: **MDHG** …         if not self.buildings: raise ValueError("MDHGHashTable has no buildings initialized.")
L16480:   - L452: **AGRM** …         Value should be (actual_value, metadata_dict) for AGRM integration.
L16481:   - L461: **MDHG** …             if not self.buildings: raise RuntimeError("MDHG Hash Table has no buildings.")
```

## 11. self.in_sample: bool = in_sample  # Will likely always be False in this version

**Source:** System_AGRM_CMPLX_DeepReview_Part8_version_120825_022806.md:L5571 [UNK/code]
```text
L5570:         self.permutation: tuple = permutation
L5571:         self.in_sample: bool = in_sample  # Will likely always be False in this version
L5572:         self.used_count: int = 0  # How many times this permutation has been used
```

**Source:** System_AGRM_CMPLX_DeepReview_Part10_version_120825_022806.md:L20765 [UNK/code]
```text
L20764:         self.permutation: tuple = permutation
L20765:         self.in_sample: bool = in_sample  # Will likely always be False in this version
L20766:         self.used_count: int = 0  # How many times this permutation has been used
```

**Source:** System_AGRM_CMPLX_DeepReview_Part13_version_120825_022806.md:L12580 [UNK/code]
```text
L12579:         self.permutation: tuple = permutation
L12580:         self.in_sample: bool = in_sample  # Will likely always be False in this version
L12581:         self.used_count: int = 0  # How many times this permutation has been used
```

## 12. if perm_hash not in eput:  # Should always be the case, but check anyway

**Source:** System_AGRM_CMPLX_DeepReview_Part8_version_120825_022806.md:L8159 [UNK/code]
```text
L8158:             perm_string = "".join(str(x) for x in best_candidate_perm)
L8159:             if perm_hash not in eput:  # Should always be the case, but check anyway
L8160:                 eput[perm_hash] = PermutationData(best_candidate_perm, in_sample=False, creation_method="dynamic_generation")
```

**Source:** System_AGRM_CMPLX_DeepReview_Part13_version_120825_022806.md:L11577 [UNK/code]
```text
L11576:             perm_string = "".join(str(x) for x in best_candidate_perm)
L11577:             if perm_hash not in eput:  # Should always be the case, but check anyway
L11578:                 eput[perm_hash] = PermutationData(best_candidate_perm, in_sample=False, creation_method="dynamic_generation")
```

**Source:** System_AGRM_CMPLX_DeepReview_Part15_version_120825_022806.md:L16887 [UNK/code]
```text
L16886:             perm_string = "".join(str(x) for x in best_candidate_perm)
L16887:             if perm_hash not in eput:  # Should always be the case, but check anyway
L16888:                 eput[perm_hash] = PermutationData(best_candidate_perm, in_sample=False, creation_method="dynamic_generation")
```

## 13. num_iterations = 1000 #Should be enough.

**Source:** System_AGRM_CMPLX_DeepReview_Part8_version_120825_022806.md:L10659 [UNK/code]
```text
L10658:     hypothetical_prodigal_generation_count = 50  # n-1 is fast
L10659:     num_iterations = 1000 #Should be enough.
L10660:     layout_k_values = [n - 2, n - 3]
```

**Source:** System_AGRM_CMPLX_DeepReview_Part13_version_120825_022806.md:L13211 [UNK/code]
```text
L13210:     hypothetical_prodigal_generation_count = 50  # n-1 is fast
L13211:     num_iterations = 1000 #Should be enough.
L13212:     layout_k_values = [n - 2, n - 3]
```

**Source:** System_AGRM_CMPLX_DeepReview_Part15_version_120825_022806.md:L18521 [UNK/code]
```text
L18520:     hypothetical_prodigal_generation_count = 50  # n-1 is fast
L18521:     num_iterations = 1000 #Should be enough.
L18522:     layout_k_values = [n - 2, n - 3]
```

## 14. raise ValueError("target_load must be in (0,1)")

**Source:** System_AGRM_CMPLX_DeepReview_Part32_version_120825_022806.md:L145 [UNK/code]
```text
L144:     if not (0.0 < cfg.target_load < 1.0):
L145:         raise ValueError("target_load must be in (0,1)")
L146:     if cfg.promote_hits < 1:
```

**Source:** System_AGRM_CMPLX_DeepReview_Part33_version_120825_022806.md:L2122 [UNK/code]
```text
L2121:     if not (0.0 < cfg.target_load < 1.0):
L2122:         raise ValueError("target_load must be in (0,1)")
L2123:     if cfg.promote_hits < 1:
```

**Source:** System_AGRM_CMPLX_DeepReview_Part34_version_120825_022806.md:L2221 [UNK/code]
```text
L2220:     if not (0.0 < cfg.target_load < 1.0):
L2221:         raise ValueError("target_load must be in (0,1)")
L2222:     if cfg.promote_hits < 1:
```

## 15. "user_context_lock": "YOU_SHELL must be set",

**Source:** System_Novel_Items_version_120825_022315.json:L77 [UNK/code]
```text
L76:       "examples": [
L77:         "SAP = Sentinel, Arbiter, Porter and is the governance layer that operates in line with the controller at highest level and aggisned levels as dictated by design notes in documents(it should be mentioned in multiple)",
L78:         "scanned the entire corpus for SAP (Sentinel/Arbiter/Porter),",
```

**Source:** System_AGRM_CMPLX_DeepReview_Part5_version_120825_022806.md:L4277 [UNK/code]
```text
L4276:     "human_proof_gate": true,
L4277:     "user_context_lock": "YOU_SHELL must be set",
L4278:     "drift_free_mode": "YES",
```

**Source:** System_AGRM_CMPLX_DeepReview_Part6_version_120825_022806.md:L15490 [UNK/code]
```text
L15489:       },
L15490:       "description": "Axis registry fields. Additional axis dims permitted but must be strings."
L15491:     },
```

## 16. - `module`: import path (must be importable from CODE_UNDER_TEST)

**Source:** System_AGRM_CMPLX_DeepReview_Part9_version_120825_022806.md:L4809 [UNK/code]
```text
L4808: - `family`: for governance and metrics
L4809: - `module`: import path (must be importable from CODE_UNDER_TEST)
L4810: - `func`: attribute in module to call
```

**Source:** System_AGRM_CMPLX_DeepReview_Part11_version_120825_022806.md:L7127 [UNK/code]
```text
L7126: - `family`: for governance and metrics
L7127: - `module`: import path (must be importable from CODE_UNDER_TEST)
L7128: - `func`: attribute in module to call
```

**Source:** README.md:L26 [UNK/code]
```text
L25: scripts/             — helper install and run scripts
L26: 
L27: reports/             — outputs of runs
```

## 17. No, Deploy to Test is JUST a method, and you should be focused on the original goal of the conversation review, gathering all data mentioned by either me or you, and listing it so we can continue working. This includes details about variation generation, sandbox environment setup, concurrent deployment, data collection, comparative analysis, test cases, expected outcomes, and acceptance criteria related to this specific concurrency testing method. You should be using all available resources (past conversations, documentation, etc.) to gather this information and document it in a structured way. The goal is to have a comprehensive understanding of the "Deploy to Test" method so we can proceed with further development and testing.

**Source:** System_AGRM_CMPLX_DeepReview_Part35_version_120825_022806.md:L3720 [UNK/prose_guide]
```text
L3719: 
L3720: No, Deploy to Test is JUST a method, and you should be focused on the original goal of the conversation review, gathering all data mentioned by either me or you, and listing it so we can continue working.  This includes details about variat
L3721: 
```

**Source:** System_AGRM_CMPLX_DeepReview_Part36_version_120825_022806.md:L16139 [UNK/prose_guide]
```text
L16138: 
L16139: No, Deploy to Test is JUST a method, and you should be focused on the original goal of the conversation review, gathering all data mentioned by either me or you, and listing it so we can continue working.  This includes details about variat
L16140: 
```

**Source:** document scanning methods.txt:L2255 [DU/other]
```text
L2254: 
L2255: No, Deploy to Test is JUST a method, and you should be focused on the original goal of the conversation review, gathering all data mentioned by either me or you, and listing it so we can continue working.  This includes details about variat
L2256: 
```

## 18. go ahead and proceed. i beleive you should be able to batch this entire job, as most code you find will be intended to work with any other

**Source:** System_AGRM_CMPLX_DeepReview_Part38_version_120825_022806.md:L10820 [UNK/other]
```text
L10819: You said:
L10820: go ahead and proceed. i beleive you should be able to batch this entire job, as most code you find will be intended to work with any other
L10821: 
```

**Source:** initial SNAP build log with GPT5.txt:L110 [UNK/log]
```text
L109: You said:
L110: go ahead and proceed. i beleive you should be able to batch this entire job, as most code you find will be intended to work with any other
L111: 
```

**Source:** initial SNAP build log.txt:L110 [UNK/log]
```text
L109: You said:
L110: go ahead and proceed. i beleive you should be able to batch this entire job, as most code you find will be intended to work with any other
L111: 
```

## 19. I didn’t find a dedicated test harness within the AGRM set. We should create a small integration test (runtime controller smoke test + candidate generation + hash invariants) and wire it to pytest or a simple script.

**Source:** System_AGRM_CMPLX_DeepReview_Part38_version_120825_022806.md:L10967 [UNK/other]
```text
L10966: 
L10967: I didn’t find a dedicated test harness within the AGRM set. We should create a small integration test (runtime controller smoke test + candidate generation + hash invariants) and wire it to pytest or a simple script.
L10968: 
```

**Source:** initial SNAP build log with GPT5.txt:L257 [UNK/log]
```text
L256: 
L257: I didn’t find a dedicated test harness within the AGRM set. We should create a small integration test (runtime controller smoke test + candidate generation + hash invariants) and wire it to pytest or a simple script.
L258: 
```

**Source:** initial SNAP build log.txt:L257 [UNK/log]
```text
L256: 
L257: I didn’t find a dedicated test harness within the AGRM set. We should create a small integration test (runtime controller smoke test + candidate generation + hash invariants) and wire it to pytest or a simple script.
L258: 
```

## 20. these will be what are used to do any logic gating onto how what where when and why the modules and tools get used. it must be VERY smart and able to make accurate decisions on what and why to use any and all parts

**Source:** System_AGRM_CMPLX_DeepReview_Part38_version_120825_022806.md:L11147 [UNK/other]
```text
L11146: 
L11147: these will be what are used to do any logic gating onto how what where when and why the modules and tools get used. it must be VERY smart and able to make accurate decisions on what and why to use any and all parts
L11148: 
```

**Source:** initial SNAP build log with GPT5.txt:L437 [UNK/log]
```text
L436: 
L437: these will be what are used to do any logic gating onto how what where when and why the modules and tools get used. it must be VERY smart and able to make accurate decisions on what and why to use any and all parts
L438: 
```

**Source:** initial SNAP build log.txt:L437 [UNK/log]
```text
L436: 
L437: these will be what are used to do any logic gating onto how what where when and why the modules and tools get used. it must be VERY smart and able to make accurate decisions on what and why to use any and all parts
L438: 
```

## 21. Understood. The controller needs universal reach and must apply smart governance—driving what/why/when/how every module is used.

**Source:** System_AGRM_CMPLX_DeepReview_Part38_version_120825_022806.md:L11154 [UNK/other]
```text
L11153: 
L11154: Understood. The controller needs universal reach and must apply smart governance—driving what/why/when/how every module is used.
L11155: 
```

**Source:** initial SNAP build log with GPT5.txt:L444 [UNK/log]
```text
L443: 
L444: Understood. The controller needs universal reach and must apply smart governance—driving what/why/when/how every module is used.
L445: 
```

**Source:** initial SNAP build log.txt:L444 [UNK/log]
```text
L443: 
L444: Understood. The controller needs universal reach and must apply smart governance—driving what/why/when/how every module is used.
L445: 
```

## 22. it should be in one of these files

**Source:** System_AGRM_CMPLX_DeepReview_Part38_version_120825_022806.md:L11264 [UNK/other]
```text
L11263: 
L11264: it should be in one of these files
L11265: 
```

**Source:** initial SNAP build log with GPT5.txt:L554 [UNK/log]
```text
L553: 
L554: it should be in one of these files
L555: 
```

**Source:** initial SNAP build log.txt:L554 [UNK/log]
```text
L553: 
L554: it should be in one of these files
L555: 
```

## 23. All functional modules must be decoupled yet fully accessible by the controller.

**Source:** System_AGRM_CMPLX_DeepReview_Part38_version_120825_022806.md:L11977 [UNK/other]
```text
L11976: 
L11977: All functional modules must be decoupled yet fully accessible by the controller.
L11978: 
```

**Source:** initial SNAP build log with GPT5.txt:L1267 [UNK/log]
```text
L1266: 
L1267: All functional modules must be decoupled yet fully accessible by the controller.
L1268: 
```

**Source:** initial SNAP build log.txt:L1267 [UNK/log]
```text
L1266: 
L1267: All functional modules must be decoupled yet fully accessible by the controller.
L1268: 
```

## 24. if not isinstance(obj, dict): raise TypeError("SNAP objects must be dicts")

**Source:** System_AGRM_CMPLX_DeepReview_Part32_version_120825_022806.md:L1061 [UNK/code]
```text
L1060:     def save(self, key: str, obj: Dict[str, Any]) -> str:
L1061:         if not isinstance(obj, dict): raise TypeError("SNAP objects must be dicts")
L1062:         kind = obj.get("kind", "")
```

**Source:** System_AGRM_CMPLX_DeepReview_Part33_version_120825_022806.md:L21619 [UNK/code]
```text
L21618:     def save(self, key: str, obj: Dict[str, Any]) -> str:
L21619:         if not isinstance(obj, dict): raise TypeError("SNAP objects must be dicts")
L21620:         kind = obj.get("kind", "")
```

**Source:** System_AGRM_CMPLX_DeepReview_Part34_version_120825_022806.md:L465 [UNK/code]
```text
L464:     def save(self, key: str, obj: Dict[str, Any]) -> str:
L465:         if not isinstance(obj, dict): raise TypeError("SNAP objects must be dicts")
L466:         kind = obj.get("kind", "")
```

## 25. Everything reviewed must be given a full indepth summary.

**Source:** System_AGRM_CMPLX_DeepReview_Part35_version_120825_022806.md:L8269 [UNK/prose_guide]
```text
L8268: 
L8269: Everything reviewed must be given a full indepth summary.
L8270: 
```

**Source:** System_AGRM_CMPLX_DeepReview_Part37_version_120825_022806.md:L3832 [UNK/code]
```text
L3831: 
L3832: Everything reviewed must be given a full indepth summary.
L3833: 
```

**Source:** session log of final build .txt:L5 [UNK/log]
```text
L4: 
L5: Everything reviewed must be given a full indepth summary.
L6: 
```

## 26. And subjects or items used as keywords must allow for room to expand to like and Sinclair other topics. Do not focus on a small set. All the work is interconnected. And different versions of the same program. The naming in the files and folders themselves help give you the timeliness.

**Source:** System_AGRM_CMPLX_DeepReview_Part35_version_120825_022806.md:L8271 [UNK/prose_guide]
```text
L8270: 
L8271: And subjects or items used as keywords must allow for room to expand to like and Sinclair other topics. Do not focus on a small set. All the work is interconnected. And different versions of the same program. The naming in the files and fol
L8272: 
```

**Source:** System_AGRM_CMPLX_DeepReview_Part37_version_120825_022806.md:L3834 [UNK/code]
```text
L3833: 
L3834: And subjects or items used as keywords must allow for room to expand to like and Sinclair other topics. Do not focus on a small set. All the work is interconnected. And different versions of the same program. The naming in the files and fol
L3835: 
```

**Source:** session log of final build .txt:L7 [UNK/log]
```text
L6: 
L7: And subjects or items used as keywords must allow for room to expand to like and Sinclair other topics. Do not focus on a small set. All the work is interconnected. And different versions of the same program. The naming in the files and fol
L8: 
```

## 27. start back on the first zip, and treeat it just like you did the original zip i gave you, and do that same thing for all of them, individual deep dives, not being informed by the first review you did and its subject matter. everything should be treated as fresh material.

**Source:** System_AGRM_CMPLX_DeepReview_Part35_version_120825_022806.md:L8942 [UNK/prose_guide]
```text
L8941: 
L8942: start back on the first zip, and treeat it just like you did the original zip i gave you, and do that same thing for all of them, individual deep dives, not being informed by the first review you did and its subject matter. everything shoul
L8943: 
```

**Source:** System_AGRM_CMPLX_DeepReview_Part37_version_120825_022806.md:L4505 [UNK/code]
```text
L4504: 
L4505: start back on the first zip, and treeat it just like you did the original zip i gave you, and do that same thing for all of them, individual deep dives, not being informed by the first review you did and its subject matter. everything shoul
L4506: 
```

**Source:** session log of final build .txt:L678 [UNK/log]
```text
L677: 
L678: start back on the first zip, and treeat it just like you did the original zip i gave you, and do that same thing for all of them, individual deep dives, not being informed by the first review you did and its subject matter. everything shoul
L679: 
```

## 28. are there not text or word documents? only code? if that is the case then that works good. if there are text and word documents, you should treat them in the same way and with the same gloves.

**Source:** System_AGRM_CMPLX_DeepReview_Part35_version_120825_022806.md:L9681 [UNK/prose_guide]
```text
L9680: You said:
L9681: are there not text or word documents? only code? if that is the case then that works good. if there are text and word documents, you should treat them in the same way and with the same gloves.
L9682: 
```

**Source:** System_AGRM_CMPLX_DeepReview_Part37_version_120825_022806.md:L5244 [UNK/code]
```text
L5243: You said:
L5244: are there not text or word documents? only code? if that is the case then that works good. if there are text and word documents, you should treat them in the same way and with the same gloves.
L5245: 
```

**Source:** session log of final build .txt:L1417 [UNK/log]
```text
L1416: You said:
L1417: are there not text or word documents? only code? if that is the case then that works good. if there are text and word documents, you should treat them in the same way and with the same gloves.
L1418: 
```

## 29. You should be doing both code and doc deep passes for all zips

**Source:** System_AGRM_CMPLX_DeepReview_Part35_version_120825_022806.md:L10688 [UNK/prose_guide]
```text
L10687: You said:
L10688: You should be doing both code and doc deep passes for all zips
L10689: 
```

**Source:** System_AGRM_CMPLX_DeepReview_Part37_version_120825_022806.md:L6251 [UNK/code]
```text
L6250: You said:
L6251: You should be doing both code and doc deep passes for all zips
L6252: 
```

**Source:** session log of final build .txt:L2424 [UNK/log]
```text
L2423: You said:
L2424: You should be doing both code and doc deep passes for all zips
L2425: 
```

## 30. There should be another code block labeled as e8 xxx part 1 2 and 3 do those files next. And the text docs accompanying them

**Source:** System_AGRM_CMPLX_DeepReview_Part35_version_120825_022806.md:L11721 [UNK/prose_guide]
```text
L11720: You said:
L11721: There should be another code block labeled as e8 xxx part 1 2 and 3 do those files next. And the text docs accompanying them 
L11722: 
```

**Source:** System_AGRM_CMPLX_DeepReview_Part37_version_120825_022806.md:L7284 [UNK/code]
```text
L7283: You said:
L7284: There should be another code block labeled as e8 xxx part 1 2 and 3 do those files next. And the text docs accompanying them 
L7285: 
```

**Source:** session log of final build .txt:L3457 [UNK/log]
```text
L3456: You said:
L3457: There should be another code block labeled as e8 xxx part 1 2 and 3 do those files next. And the text docs accompanying them 
L3458: 
```
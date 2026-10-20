# Image-adaptation error and subgroup review

Saved selected-checkpoint validation rankings; all queries retain the complete 441-study candidate pool. Groups use report-provenance metadata, multilingual-E5 token counts including prefix, and lexical last-summary-heading visibility. No test analysis or clinical image adjudication.

| Seed | Group | n | Frozen R@1 % | Adapted R@1 % | Difference pp |
| --- | --- | ---: | ---: | ---: | ---: |
| 2026 | all | 441 | 10.6576 | 11.1111 | +0.4535 |
| 2026 | source-English | 175 | 13.4286 | 13.4286 | +0.0000 |
| 2026 | translated | 266 | 8.8346 | 9.5865 | +0.7519 |
| 2026 | within-512 | 374 | 10.8289 | 11.3636 | +0.5348 |
| 2026 | over-512 | 67 | 9.7015 | 9.7015 | +0.0000 |
| 2026 | retained-heading | 221 | 10.4072 | 12.4434 | +2.0362 |
| 2026 | lost-heading | 43 | 6.9767 | 5.8140 | -1.1628 |
| 2026 | no-heading | 177 | 11.8644 | 10.7345 | -1.1299 |
| 2027 | all | 441 | 8.9569 | 10.4308 | +1.4739 |
| 2027 | source-English | 175 | 10.5714 | 10.5714 | +0.0000 |
| 2027 | translated | 266 | 7.8947 | 10.3383 | +2.4436 |
| 2027 | within-512 | 374 | 8.9572 | 11.3636 | +2.4064 |
| 2027 | over-512 | 67 | 8.9552 | 5.2239 | -3.7313 |
| 2027 | retained-heading | 221 | 9.7285 | 10.4072 | +0.6787 |
| 2027 | lost-heading | 43 | 3.4884 | 3.4884 | +0.0000 |
| 2027 | no-heading | 177 | 9.3220 | 12.1469 | +2.8249 |
| 2028 | all | 441 | 8.8435 | 10.5442 | +1.7007 |
| 2028 | source-English | 175 | 8.2857 | 12.0000 | +3.7143 |
| 2028 | translated | 266 | 9.2105 | 9.5865 | +0.3759 |
| 2028 | within-512 | 374 | 9.3583 | 10.6952 | +1.3369 |
| 2028 | over-512 | 67 | 5.9701 | 9.7015 | +3.7313 |
| 2028 | retained-heading | 221 | 8.3710 | 10.6335 | +2.2624 |
| 2028 | lost-heading | 43 | 2.3256 | 6.9767 | +4.6512 |
| 2028 | no-heading | 177 | 11.0169 | 11.2994 | +0.2825 |

These are post-selection descriptive slices, with small overlapping groups and no multiplicity or patient-cluster correction. Language describes report provenance, not an independent language detector. Summary visibility is not a clinical completeness assessment.

## Twenty worst seed-2026 direction/query pairs

Sorted by adapted expected positive rank descending, then UID/direction. Full paired report text is retained in the JSON artifact for review.

| Pair | Direction | Adapted rank | Frozen rank | Tokens | Summary |
| --- | --- | ---: | ---: | ---: | --- |
| 1 | image_to_text | 307.0 | 305.0 | 347 | no-heading |
| 2 | image_to_text | 229.0 | 347.0 | 418 | no-heading |
| 3 | text_to_image | 211.0 | 199.0 | 347 | no-heading |
| 4 | text_to_image | 192.0 | 337.0 | 418 | no-heading |
| 5 | image_to_text | 192.0 | 170.0 | 252 | retained-heading |
| 6 | image_to_text | 148.0 | 186.0 | 115 | retained-heading |
| 7 | text_to_image | 148.0 | 118.0 | 252 | retained-heading |
| 8 | text_to_image | 141.0 | 79.0 | 685 | lost-heading |
| 9 | text_to_image | 131.0 | 214.0 | 153 | retained-heading |
| 10 | text_to_image | 126.0 | 49.0 | 608 | lost-heading |
| 11 | image_to_text | 123.0 | 199.0 | 153 | retained-heading |
| 12 | text_to_image | 118.0 | 197.0 | 115 | retained-heading |
| 13 | image_to_text | 118.0 | 53.0 | 685 | lost-heading |
| 14 | image_to_text | 111.0 | 59.0 | 261 | retained-heading |
| 15 | text_to_image | 109.0 | 64.0 | 334 | retained-heading |
| 16 | image_to_text | 104.0 | 45.0 | 297 | no-heading |
| 17 | image_to_text | 100.0 | 79.0 | 591 | no-heading |
| 18 | image_to_text | 100.0 | 233.0 | 290 | retained-heading |
| 19 | image_to_text | 97.0 | 52.0 | 273 | no-heading |
| 20 | image_to_text | 97.0 | 54.0 | 608 | lost-heading |

## Review findings

The translated group gains +0.7519, +2.4436 and +0.3759 pp across seeds; source-English gains 0, 0 and +3.7143 pp. Long reports change by 0, −3.7313 and +3.7313 pp. These slices do not establish a consistent long-report improvement. Training contains 1,401 source-English and 2,123 translated reports; validation contains 175 and 266 respectively. Provenance is the appropriate discriminator: the older source-language field is mostly `und` even for retained English reports.

Reading all twenty paired reports shows unresolved exact-pair errors, with both shared report templates and differing findings. Pair numbers refer to the table and ordered JSON list. In text→image rows, the comparison text is the retrieved image's paired report, not an interpretation of its pixels.

- Pairs 1/3: the query describes ACL/MCL and meniscal tears; retrieved reports emphasize different degenerative changes. Pair 1 also switches explicitly stated right/left laterality.
- Pairs 2/4: the proximal tibial spur and associated findings are not preserved in the retrieved descriptions; pair 2 retrieves a large postoperative nerve-sheath ganglion description.
- Pairs 5/7: focal severe patellofemoral cartilage disease with intact menisci retrieves either extensive medial-compartment disease with tears or a similarly formatted report denying cartilage lesions.
- Pairs 6/12: a small medial meniscal tear retrieves a normal report or a similar template describing a more extensive tear.
- Pairs 8/13: the same multi-compartment cartilage query retrieves a same-template report with an ACL tear or another cartilage report with differing compartment emphasis. The retrieved report in pair 8 itself has medial/lateral meniscus wording inconsistency; no label correction is inferred.
- Pairs 9/11: a medial tear and anterior ganglion query retrieves a normal report or a bone-bruise report. The query's own conclusion omits the described medial tear, illustrating report uncertainty.
- Pairs 10/20: a small medial tear retrieves either a nearly identical normal template or a patellofemoral OA description.
- Pairs 14/15: tear versus degeneration distinctions and the described lesion/cyst locations are not preserved.
- Pairs 16/19: more extensive degenerative findings retrieve the same milder report with normal menisci/cartilage wording.
- Pair 17: a large anterior/medial cystic mass retrieves a posterior Baker-cyst description, with right/left disagreement.
- Pair 18: a tear and cartilage-damage report retrieves a near-normal report despite improving from rank 233 to 100.

These observations support continuing the already frozen text-feature comparison. They do not identify whether image representation, text representation, translation, or source pairing caused an error, and do not justify treating these pairs as clinically equivalent positives. No cohort, label, report, objective, or selection rule is changed.

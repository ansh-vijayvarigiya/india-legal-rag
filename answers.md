# Answer evaluation

Judged by `qwen/qwen3.8-27b`, a different model from the one that wrote the answers.

**16 of 18 answerable questions answered correctly, 0 partly. 2 of 2 unanswerable questions correctly refused. Of 20 answers judged, 0 stated something the passages do not support and 0 were supported only in part.**

| qid | category | right? | backed up? | refused? |
| --- | --- | --- | --- | --- |
| `q001` | single_clause | yes | yes | no |
| `q002` | single_clause | yes | yes | no |
| `q003` | single_clause | yes | yes | no |
| `q004` | single_clause | yes | yes | no |
| `q005` | single_clause | yes | yes | no |
| `q006` | single_clause | yes | yes | no |
| `q007` | single_clause | yes | yes | no |
| `q008` | single_clause | no | n/a | yes |
| `q009` | single_clause | yes | yes | no |
| `q010` | single_clause | yes | yes | no |
| `q011` | multi_clause | yes | yes | no |
| `q012` | multi_clause | yes | yes | no |
| `q013` | multi_clause | no | n/a | yes |
| `q014` | multi_clause | yes | yes | no |
| `q015` | cross_document | yes | yes | no |
| `q016` | cross_document | yes | yes | no |
| `q017` | temporal | yes | yes | no |
| `q018` | temporal | yes | yes | no |
| `q019` | unanswerable | no | n/a | yes |
| `q020` | unanswerable | no | n/a | yes |

### q001 · single_clause

**Question.** What is the minimum proportion of the required cash reserve ratio that scheduled banks must maintain on all days of the reporting fortnight?

**Answer.** The circular states that scheduled banks must keep **at least 90 % of the required Cash Reserve Ratio (CRR) on every day of the reporting fortnight** — the daily CRR may be lower than the prescribed rate, provided the average over the fortnight is not below the RBI‑prescribed level [1].  

- Effective from the fortnight beginning 16 October 2026, this minimum daily proportion is being raised to **99 % of the required CRR** [1].  

Thus, the current minimum is 90 % (rising to 99 % from the specified date).

**Verdict.** right: **yes** · backed up: **yes** · refused: **no**

**Why the judge says so.** The answer correctly identifies the minimum proportion as 90% (rising to 99% from October 16, 2026) based on Passage [1].

### q002 · single_clause

**Question.** What is the revised Bank Rate announced in the Monetary Policy Statement of October 07, 2026, and by how much was it raised?

**Answer.** **Revised Bank Rate:** 5.75 percent (per annum)  

**Increase:** Raised by 25 basis points (0.25 percentage point) from the previous rate of 5.50 percent.

- The Monetary Policy Statement dated 7 October 2026 announced that the Bank Rate would be increased by 25 bps, moving it from 5.50 % to 5.75 % with immediate effect[2].

**Verdict.** right: **yes** · backed up: **yes** · refused: **no**

**Why the judge says so.** The answer correctly identifies the revised Bank Rate as 5.75% and the increase as 25 basis points from 5.50%, which is explicitly stated in passage [2].

### q003 · single_clause

**Question.** Which earlier circular does the October 01, 2026 circular on online submission of Form A2 refer to, and what must Authorised Dealers frame?

**Answer.** The October 1 2026 circular on online submission of Form A2 refers to **A.P. (DIR Series) Circular No. 12 dated 3 July 2024** [1].

It requires Authorised Dealers (Category I banks and Category II entities) to **frame appropriate internal guidelines** for facilitating outward foreign‑exchange remittances through online or physical submission of Form A2, and these guidelines must be **approved by the Dealer’s Board or a Board‑delegated committee** [1].

**Verdict.** right: **yes** · backed up: **yes** · refused: **no**

**Why the judge says so.** The answer correctly identifies the referenced circular (A.P. (DIR Series) Circular No. 12 dated July 03, 2024) and accurately states that Authorised Dealers must frame appropriate internal guidelines approved by their Board or a delegated committee, as supported by Passage [1].

### q004 · single_clause

**Question.** Who may a bank's Board delegate the monitoring of the progress of cassette swap implementation in ATMs to?

**Answer.** The Board may delegate monitoring of the cassette‑swap implementation to **a Committee of Executives of the bank**[1].  

- The delegation is a partial modification of paragraph 3 of the earlier circular.  
- The Committee of Executives will be responsible for tracking progress on the ATM cassette‑swap programme.  

(Only the information in passage [1] addresses this question.)

**Verdict.** right: **yes** · backed up: **yes** · refused: **no**

**Why the judge says so.** The answer correctly identifies that the Board may delegate monitoring to a Committee of Executives, which is explicitly stated in passage [1].

### q005 · single_clause

**Question.** Which provision of the Commercial Banks (Miscellaneous) Directions, 2025 does the amendment on deferment of option premium refer to?

**Answer.** The amendment on deferment of option premium refers to **Sub‑section D of Chapter IV** of the Reserve Bank of India (Commercial Banks – Miscellaneous) Directions, 2025[1].

**Verdict.** right: **yes** · backed up: **yes** · refused: **no**

**Why the judge says so.** The answer correctly identifies Sub-section D of Chapter IV as the provision referenced in the amendment regarding deferment of option premium, which is explicitly stated in passage [1].

### q006 · single_clause

**Question.** Which entity's entry was amended in the UNSC 1267/1989 ISIL (Da'esh) and Al-Qaida Sanctions List, and under what reference number is it listed?

**Answer.** The entry that was amended is **Khatiba Jama`at al‑Tawhid wal‑Jihad (KTJ)**, which appears in the sanctions list under the reference **QDe.168**[3]. This entry was originally listed on 7 Mar 2022 and was amended on 28 Sep 2026.

**Verdict.** right: **yes** · backed up: **yes** · refused: **no**

**Why the judge says so.** The answer correctly identifies the entity as Khatiba Jama`at al-Tawhid wal-Jihad (KTJ) with reference number QDe.168, and accurately cites the listing and amendment dates (7 Mar 2022 and 28 Sep 2026) as stated in passage [3].

### q007 · single_clause

**Question.** What is the maximum number of records permitted for screen-based submission of the NRD-CSR return?

**Answer.** The NRD‑CSR (Non‑Resident Deposits – Comprehensive Single Return) can be submitted via a screen‑based web form on the RBI’s CIMS portal, but this method is limited to **a maximum of 2,000 records per submission**[1]. This is the only limit specified for screen‑based filing; other channels (XML file upload or system‑to‑system) do not have this record cap.

**Verdict.** right: **yes** · backed up: **yes** · refused: **no**

**Why the judge says so.** The answer correctly identifies the 2,000 record limit for screen-based submission as stated in passage [1] and accurately notes that other channels do not have this specific cap mentioned in the text.

### q008 · single_clause

**Question.** Under the Differential Rate of Interest Scheme, how much finance do banks provide and at what rate of interest?

**Answer.** I could not find this in the circulars provided.

**Verdict.** right: **no** · backed up: **n/a** · refused: **yes**

**Why the judge says so.** The answer correctly refuses to answer because the provided passages discuss penal interest, derivative risk management, and the Standing Liquidity Facility, but do not contain information about the Differential Rate of Interest Scheme.

### q009 · single_clause

**Question.** For what tear dimensions must a note be sorted as unfit by a note sorting machine?

**Answer.** **Answer:** A note must be sorted as unfit when it has a tear that exceeds any of the size limits listed in Table 3:

- **Vertical tear:** width ≥ 4 mm **and** length ≥ 8 mm  
- **Horizontal tear:** width ≥ 4 mm **and** length ≥ 15 mm  
- **Diagonal tear:** width ≥ 4 mm **and** length ≥ 18 mm (measured as the rectangular projection from the tear’s peak to the edge)[1]

If a tear meets or exceeds the specified width and length for its direction, the machine classifies the note as unfit.

**Verdict.** right: **yes** · backed up: **yes** · refused: **no**

**Why the judge says so.** The answer correctly identifies the specific width and length thresholds for vertical, horizontal, and diagonal tears from Table 3 in the passages, accurately reflecting the criteria for sorting notes as unfit.

### q010 · single_clause

**Question.** What must note sorting machines do with notes of a series that the Reserve Bank has decided to phase out?

**Answer.** **Answer:**  
Note‑sorting machines must treat every note belonging to a series that the Reserve Bank of India (RBI) has decided to phase out as **unfit for circulation**, sorting them as unfit regardless of their physical condition.  

**Details from the circulars:**  
- The RBI phases out certain series of notes from time to time; once a series is declared phased‑out, the machines must **sort all those notes as unfit**, even if the notes are otherwise clean or undamaged[3].  

Thus, any note of a phased‑out series is automatically classified as unfit by the machine.

**Verdict.** right: **yes** · backed up: **yes** · refused: **no**

**Why the judge says so.** The answer correctly states that machines must sort all notes of a phased-out series as unfit irrespective of their physical condition, which is explicitly supported by passage [3] section 2.3.

### q011 · multi_clause

**Question.** When must a repaired note be sorted as suspect or reject, and by what date must note sorting machines that have been discontinued from production be phased out?

**Answer.** **Repaired notes** – A note that has been repaired (joined with tape, paper, glue, etc.) must be sorted as *suspect / reject* when any of the following is true[1]  

- The repaired area is larger than 100 mm², **or**  
- The extraneous material is ≥ 50 µm thick, **or**  
- Its width is ≥ 10 mm, **or**  
- Its length is ≥ 10 mm.  

**Discontinued‑from‑production sorting machines** – Banks must phase out any Note Sorting Machine (NSM) that has been discontinued from production **on or before 30 June 2027**[2].

**Verdict.** right: **yes** · backed up: **yes** · refused: **no**

**Why the judge says so.** The answer accurately lists the four specific criteria for sorting repaired notes as suspect/reject and correctly identifies the June 30, 2027 deadline for phasing out discontinued NSMs, both of which are explicitly supported by the provided passages.

### q012 · multi_clause

**Question.** What monitoring arrangement must a bank's Head Office put in place for the flow of credit to SC/ST beneficiaries, and how must advances to SCs and STs be reported?

**Answer.** **Monitoring arrangement:**  
- The bank’s Head Office must create a **special cell** that tracks the flow of credit to Scheduled Caste (SC) and Scheduled Tribe (ST) beneficiaries, gathers branch‑level data, consolidates it and forwards the required returns to the Reserve Bank of India (RBI) and the Government[2].  

**Reporting of advances:**  
- The Head Office must **periodically review** the credit extended to SCs/STs using the returns and other branch data, and any major year‑on‑year gap or variation must be reported to the Board or the committee delegated by the Board[2].  
- Additionally, the bank should **review measures to enhance credit flow on a quarterly basis** and report progress, including field‑visit findings, to senior officers at the Head Office/Controlling Offices[2].

**Verdict.** right: **yes** · backed up: **yes** · refused: **no**

**Why the judge says so.** The answer accurately reflects the requirements for a special cell at the Head Office and the specific reporting/review procedures for SC/ST credit flow as detailed in passage [2].

### q013 · multi_clause

**Question.** In the Commercial Banks Prudential Norms on Capital Adequacy (Twelfth Amendment) Directions, 2026, which two paragraphs of the 2025 Directions were amended, and what does the amended text say an incurred CVA loss is based on?

**Answer.** I could not find this in the circulars provided

**Verdict.** right: **no** · backed up: **n/a** · refused: **yes**

**Why the judge says so.** The answer refuses to respond, but the provided passages explicitly state that paragraphs 85(6)(i) and 213(2)(iv)(b) were amended and define CVA in the context of the directions.

### q014 · multi_clause

**Question.** Under the October 10, 2026 Risk Management and Inter-Bank Dealings circular, what restriction applies to rebooking a cancelled foreign exchange derivative contract involving INR, and what must an Authorised Dealer obtain when offering such a contract to hedge a contracted exposure?

**Answer.** **Answer**  
- **Re‑booking restriction:** An Authorised Dealer (AD) must not allow a user to re‑book any foreign‑exchange derivative contract involving the Indian rupee (INR)—whether deliverable or non‑deliverable—that has been cancelled with any AD after these Directions were issued[1].  
- **Undertaking requirement:** When the AD offers an INR‑denominated derivative contract to a user for hedging a contracted exposure, it must obtain and retain a written undertaking from the user confirming that the same underlying exposure has not been hedged with any other AD (or, if partially hedged with multiple ADs, the amounts already booked elsewhere must be disclosed)[4].  

These are the only conditions stated in the October 10 2026 circulars.

**Verdict.** right: **yes** · backed up: **yes** · refused: **no**

**Why the judge says so.** The answer accurately identifies the prohibition on rebooking cancelled INR derivative contracts and the requirement to obtain an undertaking regarding the underlying exposure, both of which are explicitly stated in the provided passages.

### q015 · cross_document

**Question.** Following the Monetary Policy Statement of October 07, 2026, what changes were announced to the policy repo rate under the LAF, to the Bank Rate, and to the Standing Liquidity Facility for Primary Dealers?

**Answer.** **Answer:**  
- The policy repo rate under the Liquidity Adjustment Facility (LAF) was raised by 25 basis points to **5.50 %** (up from 5.25 %). [2]  
- The Bank Rate was also raised by 25 basis points to **5.75 %** (up from 5.50 %). [3]  
- The Standing Liquidity Facility for Primary Dealers (PDs) will now be provided at the revised repo rate of **5.50 %** with immediate effect. [1]  

**Details:**  
- Both the Monetary Policy Statement (MPS) of 7 Oct 2026 and the subsequent circulars state the repo rate move from 5.25 % to 5.50 % for LAF participants.  
- The same MPS triggers a Bank Rate increase from 5.50 % to 5.75 %, affecting penal interest rates linked to the Bank Rate.  
- The Standing Liquidity Facility, which offers collateralised liquidity to Primary Dealers, adopts the new repo rate of 5.50 % immediately.

**Verdict.** right: **yes** · backed up: **yes** · refused: **no**

**Why the judge says so.** The answer correctly identifies the 25 basis point increase in the policy repo rate to 5.50%, the Bank Rate to 5.75%, and the Standing Liquidity Facility rate to 5.50%, all of which are explicitly supported by the provided passages.

### q016 · cross_document

**Question.** Which amendment directions issued on October 07, 2026 change the requirement for clearing member banks or financial institutions to obtain a legal opinion, and which categories of entities do they cover?

**Answer.** **Amendment directions dated 7 October 2026 that alter the legal‑opinion requirement**

| Direction (date 7 Oct 2026) | What is changed | Types of institutions covered |
|---|---|---|
| **Reserve Bank of India (Small Finance Banks – Prudential Norms on Capital Adequacy) Sixth Amendment Directions** | Replaces paragraph 75(5)(i)(a) with a new wording that sets a **2 % risk weight** for a bank’s trade exposure to a qualifying central counterparty (QCCP) when the bank acts as a clearing member, whether for its own purposes or when offering clearing services to clients. This amendment updates the earlier requirement that a clearing member obtain a legal opinion. | **Small Finance Banks** (SFBs) that act as clearing members of a QCCP. |
| **Reserve Bank of India (Payments Banks – Prudential Norms on Capital Adequacy) Third Amendment Directions** | Replaces paragraph 52(5)(i)(a) with the same 2 % risk‑weight rule for a bank’s exposure to a QCCP, thereby revising the prior legal‑opinion requirement for clearing members. | **Payments Banks** that act as clearing members of a QCCP. |

Both directions expressly refer to “the requirement for clearing member banks to obtain legal opinion” and amend it to align with international standards, applying the new risk‑weight rule to the respective categories of banks. [1][3]

**Verdict.** right: **yes** · backed up: **yes** · refused: **no**

**Why the judge says so.** The answer correctly identifies the two specific amendment directions issued on October 7, 2026 (Small Finance Banks and Payments Banks) that modify the legal opinion requirement for clearing member banks, and accurately describes the changes and covered entities based on the provided passages.

### q017 · temporal

**Question.** As at the latest circular on Risk Management and Inter-Bank Dealings, dated October 12, 2026, for what purpose may Authorised Dealers offer non-deliverable derivative contracts involving INR to a non-resident user?

**Answer.** **Answer:**  
Authorised Dealers may offer non‑deliverable derivative contracts (NDDCs) involving Indian rupee (INR) to a non‑resident user **only for the purpose of hedging**[1].  

- The earlier unrestricted use has been replaced by a hedging‑only requirement.  
- This applies to Category‑I banks with an IFSC Banking Unit and Category‑III Standalone Primary Dealers.  

No other purposes are permitted under the latest October 12 2026 circular.

**Verdict.** right: **yes** · backed up: **yes** · refused: **no**

**Why the judge says so.** The answer correctly identifies that the purpose is limited to hedging and accurately cites the specific categories of Authorised Dealers mentioned in the October 12, 2026 circular.

### q018 · temporal

**Question.** What reserve must Authorised Dealers maintain for foreign exchange derivative contracts involving INR, and above what notional value does it apply?

**Answer.** **Reserve required:** Authorised Dealers must maintain a **Foreign Exchange Risk Reserve (FERR)** equal to **20 % of the INR‑equivalent of the notional amount** of each derivative contract.  

**Notional‑value threshold:** The reserve applies to foreign‑exchange derivative contracts involving INR whose **notional value exceeds USD 2 million (or its INR equivalent)**.  

- The reserve is to be deposited in cash with the Reserve Bank of India on a daily basis and held until the contract terminates [1].  
- The 20 % rate and USD 2 million threshold are specified for contracts undertaken for hedging current‑account transactions where the user purchases foreign currency against INR [1].

**Verdict.** right: **yes** · backed up: **yes** · refused: **no**

**Why the judge says so.** The answer correctly identifies the Foreign Exchange Risk Reserve (FERR) as 20% of the INR equivalent of the notional amount and the threshold as USD 2 million, both of which are explicitly stated in Passage [1].

### q019 · unanswerable

**Question.** What is the GST rate applicable to the supply of works contract services by a government authority in India?

**Answer.** I could not find this in the circulars provided

**Verdict.** right: **no** · backed up: **n/a** · refused: **yes**

**Why the judge says so.** The answer correctly refuses to answer because the provided passages contain no information regarding GST rates for works contract services.

### q020 · unanswerable

**Question.** What is the maximum deduction allowed under Section 80C of the Income-tax Act, 1961 for an individual assessee?

**Answer.** I could not find this in the circulars provided.

**Verdict.** right: **no** · backed up: **n/a** · refused: **yes**

**Why the judge says so.** The answer correctly refuses to answer because the provided passages concern RBI banking regulations and do not contain information about Section 80C of the Income-tax Act.

The judge is a language model, so these marks are a second opinion rather than proof, and it shares some of the blind spots of the model it is judging. It is a different model from the one that wrote the answers, it must give a reason for every mark, and every reason can be checked by hand against passages that are in this repository.

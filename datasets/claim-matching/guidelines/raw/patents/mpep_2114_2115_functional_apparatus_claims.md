# MPEP § 2114–2115 — Apparatus/Article Claims: Functional Language and Material Worked Upon
SOURCE_URL: https://www.uspto.gov/web/offices/pac/mpep/s2114.html ; https://www.uspto.gov/web/offices/pac/mpep/s2115.html
DOMAIN: patents

These two sections govern how examiners match apparatus/article claims — claims to physical devices — against prior art when the claim recites functional language, an intended use, or a material/article the device operates on rather than pure structure. Both sections turn on the same underlying principle: **apparatus claims are matched against prior art on the basis of structure, not on the basis of function, use, or the material worked upon.**

## § 2114 — Apparatus and article claims: functional language

### I. Inherency and functional limitations in apparatus claims

A claimed apparatus feature may be recited either structurally or functionally (e.g., "a wall configured to expand" rather than "a corrugated wall"). When an examiner concludes that a functional limitation is simply an inherent characteristic of a structure already disclosed by the prior art, establishing a prima facie case of anticipation or obviousness requires the examiner to **explain** why the prior art structure inherently possesses the functionally-recited limitation — a bare assertion is not enough (this dovetails with the reasoned-basis requirement of § 2112).

Once that explanation is given, the burden shifts to the applicant to prove the prior art structure does *not* in fact possess the asserted characteristic:

> "[W]here the Patent Office has reason to believe that a functional limitation asserted to be critical for establishing novelty in the claimed subject matter may, in fact, be an inherent characteristic of the prior art, it possesses the authority to require the applicant to prove that the subject matter shown to be in the prior art does not possess the characteristic relied on." (quoting *In re Swinehart*/*In re Best* line of cases as codified in MPEP 2114.)

### II. The central matching rule: "apparatus claims cover what a device *is*, not what a device *does*"

This is the single most important doctrinal sentence for matching apparatus claims to prior art:

> "Apparatus claims cover what a device *is*, not what a device *does*." (MPEP § 2114, citing *Hewlett-Packard Co. v. Bausch & Lomb Inc.*, 909 F.2d 1464, 1469, 15 USPQ2d 1525, 1528 (Fed. Cir. 1990).)

Consequences for matching:
- A recitation of the *manner or purpose* in which the apparatus is intended to be employed does **not** distinguish the claimed apparatus from a prior art apparatus that discloses all the same structural limitations, even where the prior art device is normally used differently. If every *structural* limitation is taught by the reference, an "intended use" clause does not defeat anticipation.
- Functional language in an apparatus/article claim is properly construed as **capability** language: it covers every device *capable of* performing the recited function, regardless of whether the prior art reference shows the device actually being used that way. So the matching test is: "is the prior art structure capable of performing the recited function?" — not "does the reference show the function being performed?"

### III. A prior art device performing all the recited functions still does not anticipate if the structure differs

The inverse matching failure mode: even if a prior art apparatus, in some mode, performs every function recited in the claim, that is not sufficient for anticipation if the prior art's structure differs from the claimed structure. Function is not a substitute for structural identity in a straight apparatus claim.

**Means-plus-function exception:** where a claim limitation is drafted in means-plus-function form (35 U.S.C. 112(f)), the matching standard is different — such a limitation reads on prior art structures that are the same as, or equivalent to, the corresponding structure disclosed in the claimed invention's own specification, not on any structure that merely performs the recited function.

### IV. Computer-implemented functional claim limitations

Purely functional claim language not tied to specific structure is construed to cover **all** devices capable of performing the recited function; if a prior art reference's structure inherently performs that function, it may anticipate or render obvious the claim under § 102/§ 103. Conversely, a functional limitation can also operate to *narrow* the claim — restricting it to the subset of structures actually capable of that specific performance, which can help distinguish over prior art that lacks that capability.

For claims that combine hardware and software (e.g., "a processor configured to..."), a claim may properly be construed to require both the hardware and the specific configuring software working together — meaning a bare general-purpose processor reference, without the specific configuration/algorithm, may not anticipate if the claim requires that specific algorithmic configuration. Where the specification does disclose an algorithm for a claimed computer-implemented function, that algorithm becomes part of what must be matched (this connects to the means-plus-function treatment of software claims under § 112(f)).

**Obviousness note (peripheral to matching, but often paired with it):** merely automating a known manual function, or adapting an existing process to use commonplace, well-known internet/computer technology, has been found obvious in several cases — i.e., not defeating an otherwise applicable prior art match just because the modern claim recites automation of a previously manual analog.

## § 2115 — Material or article worked upon by apparatus

This section addresses a specific and easily-overlooked failure mode in apparatus-claim matching: reciting, within an apparatus claim, the material or article that the apparatus works upon or produces. **Such recitations do not themselves add patentable weight and are not matched against the prior art as a distinguishing structural limitation.**

> "Inclusion of the material or article worked upon by a structure being claimed does not impart patentability to the claims." (MPEP § 2115.)

### Controlling cases

- **In re Otto** (CCPA 1963): claims to a hair-curler core member (and its manufacture) were not claims to a *method of curling hair using* that core. The court held that "the process is irrelevant" — including the fact that the claim recited hair being wound around the core — because the claim, properly read, was to the device itself, not the method of using it. Patentability could not rest on "a certain procedure for curling hair" or steps of a process nowhere actually claimed as such.
- **In re Young** (CCPA 1935): a machine claim for producing reinforced concrete beams recited limitations describing the concrete/reinforcement product formed by the machine. The court held that inclusion of the article formed, within the body of an apparatus claim, does not by itself make the claim patentable over prior art lacking only that recitation of the worked-upon article.
- **In re Casey** (CCPA 1967): an apparatus claim for a taping machine recited brush structures and tape-handling limitations. A prior art *perforating* machine (Kienzle) disclosed identical structure but was used for a different purpose. The court affirmed an obviousness rejection, holding that "references in claim 1 to adhesive tape handling do not expressly or impliedly require any particular structure in addition to that of Kienzle" — i.e., the tape-handling language did not add a structural limitation beyond what the prior art already disclosed. "[T]he manner or method in which such machine is to be utilized is not germane to the issue of patentability of the machine itself."

### Scope of the doctrine

This principle is specifically scoped to **claims directed to machinery/apparatus that works upon an article or material in its intended use** — i.e., it is a sub-species of the broader § 2114 "apparatus claims cover what a device is, not what it does" rule, applied specifically to recitations of the workpiece/output rather than of a use or function per se.

## Actionable matching criteria contributed by these sections

1. **Structure controls, not function or use.** When matching an apparatus/article claim against a reference, strip out (i) statements of intended use/purpose and (ii) recitations of the material/article worked upon or produced, and compare only the *structural* limitations that remain. A reference disclosing identical structure anticipates even if the prior art device is normally used for a different purpose or works on a different material.
2. **Functional language = capability test.** A functionally-recited limitation in an apparatus claim is matched by asking whether the prior art structure is *capable of* performing the recited function — not whether the reference shows the function actually being carried out.
3. **Function without structural identity is not enough.** A prior art device that in fact performs every recited function does not anticipate if its structure differs from the claimed structure (except where the claim itself is in means-plus-function form, which is matched against equivalents of the specification's structure).
4. **Inherent capability requires a reasoned basis** to attribute the functional limitation to the prior art structure (mirrors MPEP § 2112); burden then shifts to applicant/patentee to rebut.
5. **Software/hardware combination claims** may require matching both the specified hardware and its specific configuring algorithm — a generic processor reference alone is not necessarily a match.

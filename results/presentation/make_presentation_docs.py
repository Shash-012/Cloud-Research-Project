"""Build the two presentation hand-outs (ReportLab).

  results/presentation/presentation_day_guide.pdf   what to run, in which order, which files to open
  results/presentation/changes_vs_paper.pdf         3-page table: original paper | my change | why | what happened

Numbers are taken from the final tables in results/tables/*.csv (full grid, both environments) and docs/*.md, all
produced by the project's scripts. Rebuild: .venv\\Scripts\\python.exe results/presentation/make_presentation_docs.py
"""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

OUT = Path(__file__).resolve().parent
FONTS = Path(r"C:\Windows\Fonts")
pdfmetrics.registerFont(TTFont("Arial", str(FONTS / "arial.ttf")))
pdfmetrics.registerFont(TTFont("Arial-Bold", str(FONTS / "arialbd.ttf")))
pdfmetrics.registerFont(TTFont("Arial-Italic", str(FONTS / "ariali.ttf")))
pdfmetrics.registerFont(TTFont("Consolas", str(FONTS / "consola.ttf")))
pdfmetrics.registerFontFamily("Arial", normal="Arial", bold="Arial-Bold", italic="Arial-Italic")

INK, MUTED, RULE, ACCENT, SHADE = (colors.HexColor(c) for c in ("#1f2328", "#57606a", "#d0d7de", "#1f4e79",
                                                                "#f3f6f9"))


def styles(base):
    s = {
        "title": ParagraphStyle("title", fontName="Arial-Bold", fontSize=base + 9, leading=base + 13,
                                textColor=INK, spaceAfter=2),
        "sub": ParagraphStyle("sub", fontName="Arial", fontSize=base, leading=base + 3, textColor=MUTED,
                              spaceAfter=8),
        "h1": ParagraphStyle("h1", fontName="Arial-Bold", fontSize=base + 3, leading=base + 6, textColor=ACCENT,
                             spaceBefore=8, spaceAfter=4),
        "h2": ParagraphStyle("h2", fontName="Arial-Bold", fontSize=base + 1, leading=base + 4, textColor=INK,
                             spaceBefore=6, spaceAfter=2),
        "body": ParagraphStyle("body", fontName="Arial", fontSize=base, leading=base * 1.35, textColor=INK,
                               alignment=TA_LEFT, spaceAfter=3),
        "small": ParagraphStyle("small", fontName="Arial", fontSize=base - 1, leading=(base - 1) * 1.3,
                                textColor=MUTED),
        "cell": ParagraphStyle("cell", fontName="Arial", fontSize=base - 0.5, leading=(base - 0.5) * 1.25,
                               textColor=INK),
        "head": ParagraphStyle("head", fontName="Arial-Bold", fontSize=base - 0.5, leading=(base - 0.5) * 1.25,
                               textColor=colors.white),
        "code": ParagraphStyle("code", fontName="Consolas", fontSize=base - 1, leading=(base - 1) * 1.3,
                               textColor=INK, backColor=SHADE, borderPadding=(4, 5, 4, 5), spaceBefore=3,
                               spaceAfter=6),
    }
    s["bullet"] = ParagraphStyle("bullet", parent=s["body"], leftIndent=12, bulletIndent=2)
    return s


def code(s, text):
    return Paragraph(text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br/>"),
                     s["code"])


def bullets(s, items):
    return [Paragraph(i, s["bullet"], bulletText="•") for i in items]


def grid(s, header, rows, widths, zebra=True, section_rows=()):
    data = [[Paragraph(h, s["head"]) for h in header]]
    data += [[Paragraph(c, s["cell"]) for c in r] for r in rows]
    t = Table(data, colWidths=widths, repeatRows=1)
    style = [("BACKGROUND", (0, 0), (-1, 0), ACCENT), ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE), ("BOX", (0, 0), (-1, -1), 0.6, RULE),
             ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
             ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
    if zebra:
        style += [("BACKGROUND", (0, i), (-1, i), SHADE) for i in range(2, len(data), 2)]
    t.setStyle(TableStyle(style))
    return t


def footer(text):
    def draw(canvas, doc):
        canvas.saveState()
        canvas.setFont("Arial", 7.5)
        canvas.setFillColor(MUTED)
        w, _ = doc.pagesize
        canvas.drawString(doc.leftMargin, 9 * mm, text)
        canvas.drawRightString(w - doc.rightMargin, 9 * mm, f"page {doc.page}")
        canvas.restoreState()
    return draw


# --------------------------------------------------------------------------------------------- guide
def guide():
    s = styles(9.5)
    doc = SimpleDocTemplate(str(OUT / "presentation_day_guide.pdf"), pagesize=A4, leftMargin=18 * mm,
                            rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=16 * mm,
                            title="Presentation day guide", author="Shashwat Tripathi")
    W = A4[0] - 36 * mm
    f = []
    f += [Paragraph("Presentation day guide", s["title"]),
          Paragraph("Uncertainty-aware, Pareto-optimal autoscaling (extension of Pintye, Kovács &amp; Lovas, 2024). "
                    "What to run, which files to open, and the order to explain things in. Project root: "
                    "<font name='Consolas'>C:\\projects\\uq-autoscaler</font>. All commands are PowerShell, "
                    "run from the project root.", s["sub"])]

    f.append(Paragraph("1. The day before (Sunday): checklist", s["h1"]))
    f += bullets(s, [
        "<b>The experiment grid is finished</b> (1440 of 1440 runs ok, Sunday 15:45) and the final report is "
        "built, so nothing runs in the background during the talk. If you start anything long by mistake, the "
        "command below stops only the grid's processes.",
        "<b>Rehearse every command in section 3 once.</b> Each prints a JSON summary and saves its output. "
        "Rehearsal timings on this laptop: synthetic <i>proposed</i> 63 s, synthetic <i>cpu_threshold</i> 14 s, "
        "CloudSim Plus 60-step run 37 s (with the grid running at the same time; faster without it).",
        "<b>Keep the rehearsal plots as a fallback</b> (<font name='Consolas'>results\\runs\\demo\\proposed.png</font> "
        "and <font name='Consolas'>cpu_threshold.png</font>). If anything misbehaves live, show these instead.",
        "<b>Open in advance:</b> VS Code on the project folder, one PowerShell window at the project root, "
        "<font name='Consolas'>results\\reports\\final_report.pdf</font>, "
        "<font name='Consolas'>results\\presentation\\changes_vs_paper.pdf</font>, and the figures folder.",
        "Laptop plugged in, sleep disabled, notifications off."])
    f.append(code(s, "# stop the grid (only if something was started again)\n"
                     "$grid = Get-CimInstance Win32_Process |\n"
                     "  Where-Object { $_.CommandLine -match 'priority_driver|p4_01_run_grid' }\n"
                     "$grid | ForEach-Object { taskkill /PID $_.ProcessId /T /F }"))

    f.append(Paragraph("2. Order of the talk (about 15 to 20 minutes)", s["h1"]))
    talk = [
        ["1", "Problem and base paper", "2 min",
         "Autoscaling must pick the number of VMs each minute. The base paper predicts post-scaling metrics with a "
         "linear model (stage 1), predicts response time from them with one neural network (stage 2), and OPT picks "
         "the smallest pool predicted to meet the SLO.",
         "changes_vs_paper.pdf, page 1"],
        ["2", "What I changed (E1, E2, E3)", "3 min",
         "E1: non-linear stage 1 and a deep ensemble for stage 2. E2: uncertainty: a 95% chance constraint with "
         "stage-1 error propagated by Monte Carlo. E3: a Pareto layer over latency, cost and energy.",
         "changes_vs_paper.pdf, pages 1 to 3"],
        ["3", "Architecture and code tour", "3 min",
         "Walk the pipeline in the order data flows through it (file list in section 4).",
         "VS Code"],
        ["4", "Live demo 1: synthetic", "3 min",
         "Same workload and seed, two controllers. Show both plots side by side.",
         "section 3, commands A and B"],
        ["5", "Live demo 2: CloudSim Plus", "2 min",
         "The same Python models drive a Java discrete-event simulator over a socket bridge, one decision per "
         "simulated minute.",
         "section 3, command C"],
        ["6", "Results", "4 min",
         "Main table, trade-off figure, ablations, power sensitivity.",
         "final_report.pdf; figures fig3, fig4, fig5, fig9"],
        ["7", "Honest findings and limitations", "2 min",
         "Where the method did not help (section 5).",
         "final_report.pdf, Findings and Limitations"],
    ]
    f.append(grid(s, ["#", "Segment", "Time", "What to say", "What to show"], talk,
                  [7 * mm, 33 * mm, 12 * mm, 82 * mm, W - 134 * mm]))

    f.append(Paragraph("3. What to run live", s["h1"]))
    f.append(Paragraph("A and B: the method against a baseline on the same flash-crowd workload "
                       "(synthetic environment)", s["h2"]))
    f.append(code(s, ".venv\\Scripts\\python.exe scripts/p2_05_run_prototype.py --policy proposed --workload "
                     "flash_crowd --seed 1 --out results/runs/demo --figure results/runs/demo/proposed.png\n"
                     ".venv\\Scripts\\python.exe scripts/p2_05_run_prototype.py --policy cpu_threshold --workload "
                     "flash_crowd --seed 1 --out results/runs/demo --figure results/runs/demo/cpu_threshold.png"))
    f.append(KeepTogether([
        Paragraph("Each run simulates one day (1440 one-minute steps) and prints a JSON summary. "
                  "Point at three fields. Rehearsal values:", s["body"]),
        grid(s, ["Run", "violation_rate", "cost_usd", "energy_kwh"],
             [["proposed", "0.000", "22.84", "38.46"], ["cpu_threshold", "0.041", "25.33", "39.93"]],
             [40 * mm, 35 * mm, 30 * mm, 30 * mm], zebra=False)]))
    f.append(Spacer(1, 4))
    f += bullets(s, [
        "<b>What to say about the plots:</b> the top panel shows the request rate and the pool size, the bottom "
        "panel the mean response time against the 250 ms SLO. <i>proposed</i> raises the pool together with each "
        "of the three surges and stays far below the SLO. <i>cpu_threshold</i> adds one VM at a time with a "
        "cooldown, so response time shoots above the SLO at the start of each surge, and it keeps too many VMs "
        "afterwards.",
        "<b>Be ready for:</b> \"why is the pool line so jagged?\" The method changes the pool in about half of the "
        "minutes (churn), mostly by ±1; this is a known limitation (section 5).",
        "The full run log and the decision log (every candidate's bound, cost and energy) are saved in the run "
        "directory printed at the end; open <font name='Consolas'>summary.json</font> there if asked."])
    f.append(Paragraph("C: the CloudSim Plus simulator with the Python models in the loop", s["h2"]))
    f.append(code(s, ".venv\\Scripts\\python.exe scripts/p3_02_run_cloudsim.py --policy proposed --workload step "
                     "--seed 3 --steps 60 --out results/runs/demo"))
    f += bullets(s, [
        "Runs 60 simulated minutes of the step workload (37 s in rehearsal). Java simulates every request; at each "
        "minute it sends telemetry to Python, which returns the scaling action. The printout ends with "
        "<font name='Consolas'>\"status\": \"ok\"</font> and <font name='Consolas'>\"rows\": 60</font>.",
        "Open the run directory it prints: <font name='Consolas'>run_log.csv</font> (one row per minute: pool, "
        "response time, cost, energy) and <font name='Consolas'>java_stdout.txt</font>."])
    f.append(Paragraph("D (optional): the test suite", s["h2"]))
    f.append(code(s, ".venv\\Scripts\\python.exe -m pytest -q          # 121 tests, about 40 s\n"
                     "mvn -q -f cloudsim/pom.xml test                    # 12 simulator validation tests"))
    f.append(Paragraph("Use this only if asked how you know the simulator is right: the Java tests include an "
                       "M/M/1 queue check (100.07 ms against the theoretical 100 ms) and an energy cross-check "
                       "against CloudSim Plus's own power model (0.0002% difference).", s["body"]))

    f.append(Paragraph("4. Files to open during the code tour (in this order)", s["h1"]))
    files = [
        ["configs/experiments.yaml", "Policies, the seven weight vectors, the ten ablations, workloads, seeds."],
        ["src/autoscaler/models/metric_filters.py", "Base paper's metric selection (t-test and R² filters); kept."],
        ["src/autoscaler/models/metric_estimator.py", "E1, stage 1: four candidate models per metric, chosen by "
                                                      "cross-validated R²; keeps out-of-fold residuals."],
        ["src/autoscaler/models/rt_ensemble.py", "E1/E2, stage 2: five networks, each predicting a mean and a "
                                                 "variance of log response time (Gaussian NLL loss)."],
        ["src/autoscaler/uncertainty/propagation.py", "E2: Monte Carlo propagation of stage-1 error through every "
                                                      "member; total σ and the 95% bound U_k."],
        ["src/autoscaler/decision/selection.py", "E2/E3: feasibility U_k ≤ SLO, Pareto front, augmented "
                                                 "Tchebycheff choice, fallback."],
        ["src/autoscaler/decision/power.py, objectives.py", "E3: power models and the latency, cost and energy "
                                                            "objectives."],
        ["src/autoscaler/decision/policies.py", "All controllers: thresholds, OPT (base paper), proposed, "
                                                "ablations."],
        ["cloudsim/src/main/java/org/uqautoscaler/sim/ControlLoop.java", "Java side: one decision per simulated "
                                                                        "minute; RemotePolicy.java calls Python."],
        ["src/autoscaler/bridge/server.py", "Python side of the socket bridge."],
        ["scripts/p4_01_run_grid.py", "Runs the experiment grid (resumable, parallel, manifest.csv)."],
        ["scripts/p5_01 to p5_04", "Aggregation, statistics (Wilcoxon, Holm, hypervolume), figures, report."],
    ]
    f.append(grid(s, ["File", "What it shows"], files, [78 * mm, W - 78 * mm]))

    f.append(PageBreak())
    f.append(Paragraph("5. Results to show and how to frame them", s["h1"]))
    f += bullets(s, [
        "<b>Main table</b> (final_report.pdf, Main results): violation rate, p95 response time, cost and energy per "
        "workload as mean ± 95% CI over 10 seeds, for both environments. Headline (CloudSim Plus): the method's "
        "violation rate is 0.000 on three workloads and 0.012 on step, against 0.24 to 0.39 for OPT, 0.49 to "
        "0.56 for the RT threshold and 0 to 0.13 for the CPU threshold; it uses the least energy on every "
        "CloudSim workload (synthetic: 3 of 4; OPT is lower on flash_crowd).",
        "<b>Trade-off figure</b> (<font name='Consolas'>results\\figures\\fig4_tradeoff_*.png</font>): the seven "
        "weight vectors form a front; the baselines are single points. Hypervolume is highest for the method on "
        "every workload in both environments (CloudSim 0.75 to 0.97 against at most 0.45 for any baseline).",
        "<b>Ablations</b> (fig5): which components matter. Only the Pareto layer (against min-cost selection) "
        "changes violations in both environments; the linear stage 1 does in the synthetic system only. The "
        "uncertainty components never reduce violations; they make the method more expensive.",
        "<b>Power sensitivity</b> (fig9): the cost–energy trade-off appears only above ΔP/P<sub>idle</sub> ≈ 1.42, "
        "as the analysis predicted, in both environments.",
        "<b>Say plainly where it did not help</b>: in CloudSim it costs 1.3 to 2.2 USD more per run than OPT and "
        "the RT threshold (both under-provision heavily) and the CPU threshold also has zero violations on "
        "two workloads, at similar or higher cost; the uncertainty components cost money without reducing violations; churn is high; OPT "
        "oscillates, so beating OPT is a weak claim on its own.",
        "<b>If asked about significance:</b> with 10 seeds, a Holm correction across a whole table cannot reach "
        "p &lt; 0.05 (smallest attainable 0.14 for the baselines, 0.23 for the ablations), so the report also "
        "gives Holm within each environment, workload and metric, labelled as secondary."])

    f.append(Paragraph("6. Backup material if a question needs it", s["h1"]))
    f.append(grid(s, ["Question", "Open"], [
        ["How good are the models?", "final_report.pdf, Model quality; results/tables/stage1_r2*.csv, "
                                     "calibration*.csv; fig6, fig7"],
        ["Is the bound calibrated?", "fig6_calibration.png (coverage 0.970 synthetic, 0.967 CloudSim at the 95% "
                                     "level)"],
        ["Does the simulator agree with theory?", "docs/implementation_notes.md, section 4 (validation table)"],
        ["Which runs were done?", "results/runs/manifest.csv; final_report.pdf, Setup"],
        ["Can results be reproduced?", "results/tables/reproducibility.csv: 5 random runs re-run, all 5 run "
                                       "logs identical (decision logs equal up to float32 rounding, 2e-7)"],
        ["Did any runs fail?", "docs/failed_runs.md: none failed twice; 16 CloudSim attempts were retried "
                               "(bridge timeout under overload, laptop sleep)"],
        ["Why does OPT do so badly?", "docs/prototype_findings.md, \"What did not behave as expected\", item 1"],
    ], [60 * mm, W - 60 * mm]))
    doc.build(f, onFirstPage=footer("Presentation day guide"), onLaterPages=footer("Presentation day guide"))


# --------------------------------------------------------------------------------------------- changes
SYN = "<font color='#57606a'>[synthetic]</font>"
CS = "<font color='#57606a'>[CloudSim]</font>"


def changes():
    s = styles(9.3)
    doc = SimpleDocTemplate(str(OUT / "changes_vs_paper.pdf"), pagesize=landscape(A4), leftMargin=12 * mm,
                            rightMargin=12 * mm, topMargin=9 * mm, bottomMargin=13 * mm,
                            title="Changes to the base paper", author="Shashwat Tripathi")
    W = landscape(A4)[0] - 24 * mm
    widths = [0.18 * W, 0.25 * W, 0.18 * W, 0.39 * W]
    head = ["Original paper", "My change", "Why this change", "What happened with this change"]

    def section(title, note, rows):
        return [Paragraph(title, s["h1"]), Paragraph(note, s["small"]), Spacer(1, 3), grid(s, head, rows, widths)]

    f = [Paragraph("What I changed in the base paper, and what it did", s["title"]),
         Paragraph("Base paper: Pintye, Kovács &amp; Lovas (2024), two-stage response-time prediction with the OPT "
                   "scaling rule. Results are means over 10 seeds on 4 workloads in two environments, all 1440 "
                   "runs complete: synthetic (one simulated day per run) and CloudSim Plus (6 simulated hours per "
                   "run). Violation rate is the share of one-minute intervals above the 250 ms SLO; cost is USD and "
                   "energy kWh per run. Ablation differences are “full method minus ablation” on daily_ramp_burst / "
                   "flash_crowd.", s["sub"])]

    f += section("E1: Prediction", "Stage 1 predicts post-scaling metrics for each candidate action; stage 2 "
                                    "predicts response time from them.", [
        ["<b>Metric selection:</b> keep a metric if it moves differently after scale-up and scale-down "
         "(t-test) and stage 1 can estimate it well (R² &gt; 0.90).",
         "Kept as in the paper; applied separately to the synthetic and the CloudSim Plus metrics.",
         "The filters are sound: a metric that cannot be estimated only adds noise to stage 2.",
         "Synthetic: cpu_util, mem_util, disk_iops selected; queue_len rejected (best R² 0.856). CloudSim: "
         "cpu_util, net_in_kbps, throughput_rps selected; mem_util (0.80) and queue_len (0.11) rejected."],
        ["<b>Stage 1:</b> linear regression on two features per metric, [m·w/n, m·k/n] (paper Eq. 1).",
         "Per metric, choose among four models (paper linear, linear on basis features, gradient boosting, "
         "ensemble of five MLPs) by 5-fold cross-validated R²; keep out-of-fold residuals.",
         "CPU and memory respond non-linearly to adding or removing VMs. The residuals are needed for the "
         "uncertainty in E2.",
         "CPU R² 0.968 (MLP) vs 0.919 best linear " + SYN + "; CloudSim CPU 0.904 (MLP). Using the paper's linear "
         "stage 1 inside my method raises the violation rate from about 0 to <b>0.080 / 0.085</b>, the clearest "
         "ablation effect " + SYN + "; in CloudSim it changes nothing (violations ≤ 0.001). It also explains OPT's "
         "failure: the linear model predicts about 0.87 × current CPU whatever the action."],
        ["<b>Stage 2:</b> one network (two hidden layers, LeakyReLU, dropout), MSE loss, a single point "
         "prediction of response time.",
         "Deep ensemble of five such networks; each outputs a mean and a variance of log response time (Gaussian "
         "negative log-likelihood), trained on bootstrap samples with early stopping; 20-trial hyperparameter "
         "search.",
         "A decision under uncertainty needs a predictive distribution, not a point; member disagreement measures "
         "model (epistemic) uncertainty.",
         "R² on log response time 0.995 " + SYN + ", 0.908 CloudSim. One member instead of five changes almost "
         "nothing: violations equal, cost −0.04 / +0.13 USD. Stage-1 error dominates the total uncertainty, so "
         "ensemble disagreement adds little here. Same in CloudSim."],
        ["<b>Request rate</b> left out of the final model; whether it helps was left open.",
         "Tested as an ablation: a separately trained ensemble with the request rate as an extra input.",
         "Answer the paper's open question.",
         "No gain: R² 0.9947 with the rate vs 0.9951 without; outcomes within 0.15 USD and 0.03 kWh. The paper's "
         "choice is confirmed; same in CloudSim (within 0.04 USD)."],
    ])

    f.append(PageBreak())
    f += section("E2: Uncertainty", "How the controller decides whether a candidate pool size is safe.", [
        ["<b>Feasibility test:</b> point prediction below the SLO, e<super>ŷ</super> &lt; SLO.",
         "Chance constraint: U<sub>k</sub> = exp(m̄ + 1.645 σ) ≤ SLO, a one-sided 95% upper bound on response "
         "time.",
         "A point prediction close to the SLO is exceeded about half the time.",
         "Violation rate of the method 0.000 to 0.001 vs 0.003 to 0.087 (RT threshold) " + SYN + "; 0.000 to "
         "0.012 vs 0.49 to 0.56 (RT threshold), 0 to 0.13 (CPU threshold), 0.24 to 0.39 (OPT) " + CS + ". "
         "<b>But</b> without the bound violations are also about 0: the bound costs <b>+3.0 / +2.3 USD</b> and "
         "saves 1.2 / 0.4 kWh " + SYN + "; in CloudSim it costs +0.58 / +0.25 USD and uses slightly <i>more</i> "
         "energy (+0.06 / +0.04 kWh)."],
        ["<b>Stage-2 inputs treated as exact,</b> although they are stage-1 estimates.",
         "Monte Carlo propagation: 32 draws of stage-1 error (residual covariance) pushed through every ensemble "
         "member; total σ = aleatoric + epistemic + stage-1 parts.",
         "Ignoring stage-1 error makes the bound over-confident.",
         "End-to-end coverage at the 95% level: no uncertainty 0.47, member disagreement only 0.58, aleatoric + "
         "epistemic 0.91, <b>full 0.97</b> " + SYN + "; 0.967 " + CS + ". Stage-1 error is the largest part of σ "
         "(0.27 of 0.31). Effect on decisions: no violation difference, +2.2 / +1.4 USD, −0.8 / −0.2 kWh " + SYN +
         "; +0.50 / +0.22 USD and +0.07 / +0.05 kWh " + CS + "."],
        ["<b>No uncertainty model.</b>",
         "Alternative tested as ablations: a direct 95% quantile model (quantile neural network, quantile "
         "gradient boosting).",
         "A simpler way to get an upper bound, without an ensemble.",
         "End-to-end coverage 0.87 (NN) and 0.87 (GB) " + SYN + ", 0.96 and 0.77 " + CS + ", against 0.97 for the "
         "full method. Violations equal to the full method; the full method costs +2.3 / +1.6 USD more "
         + SYN + " and +0.1 to +0.3 USD more " + CS + "."],
        ["—",
         "Counterfactual check in CloudSim Plus: 50 states × 7 actions (350 runs), comparing U<sub>k</sub> with "
         "the measured outcome.",
         "Test the bound on outcomes of actions actually taken, not only on held-out rows.",
         "Coverage 0.96 at k = −3 (overload) and 1.00 at k = −2 to +3: conservative at low load, where it cannot "
         "change a decision (bound about 75 ms vs a 250 ms SLO). Accepted as a documented limitation."],
    ])

    f.append(PageBreak())
    f += section("E3: Decision", "Which pool size to choose among the safe ones.", [
        ["<b>OPT:</b> scan actions from the largest scale-in upward and take the first one predicted to meet the "
         "SLO (the cheapest); if none, the largest pool.",
         "Pareto layer: keep feasible candidates not dominated in (latency bound, cost, energy); pick one by "
         "augmented Tchebycheff with weights (w<sub>L</sub>, w<sub>C</sub>, w<sub>E</sub>); 7 weight vectors. "
         "Same fallback.",
         "Cost and energy can disagree; an operator should be able to choose the trade-off instead of always "
         "taking the cheapest pool.",
         "Hypervolume highest on every workload in both environments: <b>0.96 / 1.05 / 0.94 / 0.74</b> vs at most "
         "0.76 for any baseline " + SYN + "; <b>0.87 / 0.97 / 0.88 / 0.75</b> vs at most 0.45 " + CS + ". The "
         "cheapest-feasible rule inside my method (min-cost ablation) saves 6.8 / 5.8 USD but uses 3.8 / 2.1 kWh "
         "more and raises violations by 0.022 / 0.044 " + SYN + "; in CloudSim it saves 1.6 / 1.2 USD, uses 0.5 / "
         "0.2 kWh more and raises violations by 0.004 / <b>0.051</b>."],
        ["<b>No energy objective</b> in the decision.",
         "Energy per candidate from a server power model P(u) = P<sub>idle</sub> + (P<sub>max</sub> − "
         "P<sub>idle</sub>) h(u), h linear, square or cubic.",
         "Analysis predicts a real cost–energy trade-off only when ΔP/P<sub>idle</sub> &gt; 1/u<sub>slo</sub>² "
         "≈ 1.42.",
         "Confirmed in both environments. Share of decisions with more than one Pareto point (square model) at "
         "ratio 1 / 1.42 / 2 / 4: 0 / 0.003 / 0.65 / 1.0 " + SYN + ", 0 / 0.095 / 0.87 / 1.0 " + CS + "; linear "
         "model never, cubic always."],
        ["—",
         "Two-objective ablation: Pareto layer over cost and energy only (latency only as the constraint).",
         "Does latency as an objective add anything beyond the SLO constraint?",
         "Violations stay about 0. With latency the method costs +2.2 / +3.4 USD more and uses 0.8 / 0.7 kWh "
         "less " + SYN + ": latency as an objective buys energy, not safety. In CloudSim it costs +0.35 USD and "
         "uses slightly more energy, so it buys nothing there."],
    ])

    f += section("Evaluation", "How the comparison was made.", [
        ["<b>Baseline:</b> the paper's own OPT rule.",
         "Added two common rules, CPU threshold and response-time threshold (+1 VM per step, 3-step cooldown); OPT "
         "re-implemented exactly as specified.",
         "A fair comparison needs the rules used in practice.",
         "OPT oscillates: violation rate about 0.50 and 1238 actions per day " + SYN + ", 0.24 to 0.39 " + CS +
         ", so beating OPT means little. The method costs <b>+5.0 to +6.5 USD</b> more than the RT threshold "
         + SYN + " and +1.3 to +2.2 USD more than OPT and the RT threshold " + CS + ", which both under-provision; "
         "it is cheaper than the CPU threshold on 3 of 4 workloads " + CS + "."],
        ["<b>Evaluation setup:</b> the paper's own experiments (not reproduced here).",
         "Two environments: a fast synthetic Python system and CloudSim Plus 8.5.7 (Java, discrete-event), with "
         "models retrained on simulator data and a socket bridge to Python.",
         "Controlled, repeatable workloads; a simulator closer to real request queues.",
         "Simulator validated: M/M/1 mean 100.07 ms vs 100 ms, energy within 0.0002% of CloudSim Plus's power "
         "model (after fixing the event resolution: 1 ms gave +6.3%). All 1440 runs ok; 5 random re-runs give "
         "identical run logs."],
        ["<b>Statistics:</b> —",
         "4 workloads × 10 seeds, 95% t-intervals, paired Wilcoxon tests with Holm correction, hypervolume.",
         "Differences must hold across seeds, not in a single run.",
         "With 10 seeds, table-wide Holm cannot reach p &lt; 0.05 (minimum 0.14 baselines, 0.23 ablations). Holm "
         "per environment, workload and metric (secondary): every violation difference vs a baseline significant "
         "unless both are 0; among ablations only min-cost and linear stage 1 (synthetic) change violations."],
        ["—",
         "Reported the method's side effects.",
         "Honest reporting.",
         "High churn: 520 to 990 actions per day " + SYN + ", 148 to 251 per run " + CS + " (thresholds 11 to 155). "
         "Realised coverage down to 0.88 with weights (1,3,1) " + SYN + "; at least 0.98 " + CS + "."],
    ])
    doc.build(f, onFirstPage=footer("Changes to the base paper"), onLaterPages=footer("Changes to the base paper"))


if __name__ == "__main__":
    guide()
    changes()
    print("written:", OUT / "presentation_day_guide.pdf", OUT / "changes_vs_paper.pdf")

"""
make_report_w4.py - Export the Workshop 4 results to LaTeX.

Machine Learning - Workshop 4

Reads the JSON/CSV files written by ``train_ae.py``, ``evaluate_ae.py`` and
``analyze_latent.py`` and writes, in ``report/workshop4/generated/``:

    results.tex        one \\newcommand per number quoted in the text
    tab-recon.tex      reconstruction quality (all AE z and VAE beta runs)
    tab-anomaly.tex    cumulative comparison W1 / W2 / W4 (anomaly detection)
    tab-sweep.tex      anomaly ROC-AUC per latent size / beta and training regime
    tab-latent.tex     separability of the four representations

``report/workshop4/main.tex`` \\input's these files, so after re-running the
experiments the report is updated without editing any number by hand.

Usage
-----
    python src/make_report_w4.py
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from w4_common import GEN_DIR, RUNS_DIR, TrainConfig, load_json


def f3(v: float) -> str:
    """Format with three decimals."""
    return f"{v:.3f}"


def f4(v: float) -> str:
    """Format with four decimals."""
    return f"{v:.4f}"


def pct(v: float) -> str:
    """Format a fraction as a LaTeX percentage."""
    return f"{100 * v:.1f}\\%"


def bold_max(values: List[float], fmt=f3, minimize: bool = False) -> List[str]:
    """Format a column and bold its best value."""
    best = min(values) if minimize else max(values)
    return [f"\\textbf{{{fmt(v)}}}" if abs(v - best) < 1e-12 else fmt(v) for v in values]


def main() -> None:
    """Write every generated LaTeX file."""
    sel = load_json(RUNS_DIR / "selection.json")
    rec = load_json(RUNS_DIR / "reconstruction_results.json")
    anom = load_json(RUNS_DIR / "anomaly_results.json")
    lat = load_json(RUNS_DIR / "latent_results.json")
    train = pd.read_csv(RUNS_DIR / "training_summary.csv")
    cum = pd.read_csv(RUNS_DIR / "cumulative_results.csv")
    GEN_DIR.mkdir(parents=True, exist_ok=True)

    z, b = sel["z_star"], sel["beta_star"]
    ae_run = TrainConfig(model="ae", z_dim=z).name
    vae_run = TrainConfig(model="vae", z_dim=z, beta=b).name
    R = rec["reconstruction"]
    M = anom["main"]
    S = anom["sweep"]
    L = lat["representations"]
    B = lat["beta_study"]
    D = rec["denoising"]
    G = rec["generation"]

    def tf(key: str) -> Dict:
        return M[key]["test_f1"]

    w1 = cum[cum["workshop"] == "W1"].set_index("model")
    macros: Dict[str, str] = {
        "zstar": str(z), "betastar": f"{b:g}",
        "epochs": str(sel["epochs"]), "daenoise": f"{sel['dae_noise']:g}",
        # Reconstruction (all-rows models, test)
        "aeMSE": f4(R[ae_run]["mse"]), "aeRsq": f3(R[ae_run]["r2"]), "aeCorr": f3(R[ae_run]["mean_corr"]),
        "vaeMSE": f4(R[vae_run]["mse"]), "vaeRsq": f3(R[vae_run]["r2"]), "vaeCorr": f3(R[vae_run]["mean_corr"]),
        "aeMSEsafe": f4(R[ae_run]["mse_normal"]), "aeMSEfail": f4(R[ae_run]["mse_anomaly"]),
        "vaeMSEsafe": f4(R[vae_run]["mse_normal"]), "vaeMSEfail": f4(R[vae_run]["mse_anomaly"]),
        "aeTopAnom": pct(R[ae_run]["top1pct"]["anomaly_rate"]),
        "vaeTopAnom": pct(R[vae_run]["top1pct"]["anomaly_rate"]),
        "aeTopXdot": f"{R[ae_run]['top1pct']['mean_abs_x_dot']:.2f}",
        "vaeTopXdot": f"{R[vae_run]['top1pct']['mean_abs_x_dot']:.2f}",
        "allXdot": f"{R['overall']['mean_abs_x_dot']:.2f}",
        "aeTopX": f"{R[ae_run]['top1pct']['mean_abs_x']:.2f}",
        "vaeTopX": f"{R[vae_run]['top1pct']['mean_abs_x']:.2f}",
        "allX": f"{R['overall']['mean_abs_x']:.2f}",
        "aeTopFar": pct(R[ae_run]['top1pct']['frac_abs_x_gt_1_5']),
        "vaeTopFar": pct(R[vae_run]['top1pct']['frac_abs_x_gt_1_5']),
        "allFar": pct(R['overall']['frac_abs_x_gt_1_5']),
        "aeTopThetadot": f"{R[ae_run]['top1pct']['mean_abs_theta_dot']:.2f}",
        "vaeTopThetadot": f"{R[vae_run]['top1pct']['mean_abs_theta_dot']:.2f}",
        "allThetadot": f"{R['overall']['mean_abs_theta_dot']:.2f}",
        "testAnomRate": pct(R["overall"]["anomaly_rate"]),
        "vaeKL": f"{R[vae_run]['kl']:.2f}", "vaeActive": str(R[vae_run]["active_units"]),
        # Anomaly detection (normal-rows models, test, validation threshold)
        "aeAUC": f3(tf("AE")["roc_auc"]), "aePR": f3(tf("AE")["pr_auc"]), "aeFone": f3(tf("AE")["f1"]),
        "aeRec": f3(tf("AE")["recall"]), "aePrec": f3(tf("AE")["precision"]),
        "vaeAUC": f3(tf("VAE (recon)")["roc_auc"]), "vaePR": f3(tf("VAE (recon)")["pr_auc"]),
        "vaeFone": f3(tf("VAE (recon)")["f1"]),
        "vaeklAUC": f3(tf("VAE (recon + KL)")["roc_auc"]), "vaeklPR": f3(tf("VAE (recon + KL)")["pr_auc"]),
        "vaeklFone": f3(tf("VAE (recon + KL)")["f1"]),
        "daeAUC": f3(tf("Denoising AE")["roc_auc"]), "daePR": f3(tf("Denoising AE")["pr_auc"]),
        "supAUC": f3(tf("Supervised MLP (W2 ref.)")["roc_auc"]),
        "supPR": f3(tf("Supervised MLP (W2 ref.)")["pr_auc"]),
        "supFone": f3(tf("Supervised MLP (W2 ref.)")["f1"]),
        "gbAUC": f3(w1.loc["Gradient Boosting", "roc_auc"]), "gbPR": f3(w1.loc["Gradient Boosting", "pr_auc"]),
        "gbFone": f3(w1.loc["Gradient Boosting", "f1"]),
        "lrAUC": f3(w1.loc["Logistic Regression", "roc_auc"]), "lrPR": f3(w1.loc["Logistic Regression", "pr_auc"]),
        "aeQrec": f3(M["AE"]["test_q99"]["recall"]), "aeQprec": f3(M["AE"]["test_q99"]["precision"]),
        "vaeklQrec": f3(M["VAE (recon + KL)"]["test_q99"]["recall"]),
        "vaeklQprec": f3(M["VAE (recon + KL)"]["test_q99"]["precision"]),
        "aeAllAUC": f3(S[TrainConfig(model="ae", regime="all", z_dim=z).name]["roc_auc"]),
        # Latent space
        "silRaw": f3(L["Input space (16-D)"]["silhouette"]["balanced"]),
        "silAE": f3(L["AE"]["silhouette"]["balanced"]), "silVAE": f3(L["VAE"]["silhouette"]["balanced"]),
        "silSup": f3(L["Supervised (W2 ref.)"]["silhouette"]["balanced"]),
        "probeRaw": f3(L["Input space (16-D)"]["probe"]["roc_auc"]),
        "probeAE": f3(L["AE"]["probe"]["roc_auc"]), "probeVAE": f3(L["VAE"]["probe"]["roc_auc"]),
        "probeSup": f3(L["Supervised (W2 ref.)"]["probe"]["roc_auc"]),
        "probePRraw": f3(L["Input space (16-D)"]["probe"]["pr_auc"]),
        "probePRAE": f3(L["AE"]["probe"]["pr_auc"]), "probePRVAE": f3(L["VAE"]["probe"]["pr_auc"]),
        "probePRSup": f3(L["Supervised (W2 ref.)"]["probe"]["pr_auc"]),
        # Denoising
        "aeNoisyMSE": f4(D["AE"]["mse_noisy_input"]), "daeNoisyMSE": f4(D["Denoising AE"]["mse_noisy_input"]),
        "aeCleanMSE": f4(D["AE"]["mse_clean_input"]), "daeCleanMSE": f4(D["Denoising AE"]["mse_clean_input"]),
        "noiseVar": f4(D["identity_on_noisy_input"]),
        # Generation
        "genWvae": f3(sum(G[c]["w1_vae_over_std"] for c in ("x", "x_dot", "theta", "theta_dot")) / 4),
        "genWae": f3(sum(G[c]["w1_ae_over_std"] for c in ("x", "x_dot", "theta", "theta_dot")) / 4),
        "genConsVae": f"{G['abs_theta_consistency_mae_rad']['vae']:.3f}",
        "genConsAe": f"{G['abs_theta_consistency_mae_rad']['ae']:.3f}",
        "genBeyond": pct(G["generated_theta_beyond_limit"]["vae"]),
    }
    for bb in sel["beta_grid"]:
        tag = {0.5: "Half", 1.0: "One", 4.0: "Four"}.get(bb, str(bb).replace(".", ""))
        macros[f"active{tag}"] = str(B[f"{bb:g}"]["active_units"])
        macros[f"kl{tag}"] = f"{B[f'{bb:g}']['kl']:.2f}"
        macros[f"mse{tag}"] = f4(B[f"{bb:g}"]["test_mse"])
        macros[f"silbeta{tag}"] = f3(B[f"{bb:g}"]["silhouette"]["balanced"])
    # training rows / times
    macros["trainRowsAll"] = f"{int(train[train['regime'] == 'all']['train_rows'].iloc[0]):,}"
    macros["trainRowsNormal"] = f"{int(train[train['regime'] == 'normal']['train_rows'].iloc[0]):,}"
    for zz in sel["z_grid"]:
        tag = {2: "Two", 4: "Four", 8: "Eight"}.get(zz, str(zz))
        macros[f"aeMSEz{tag}"] = f4(R[TrainConfig(model="ae", z_dim=zz).name]["mse"])
        macros[f"aeAUCz{tag}"] = f3(S[TrainConfig(model="ae", regime="normal", z_dim=zz).name]["roc_auc"])

    with open(GEN_DIR / "results.tex", "w") as f:
        f.write("% Auto-generated by src/make_report_w4.py - do not edit by hand\n")
        for k, v in macros.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

    # ---- Table: reconstruction ------------------------------------------- #
    runs = [TrainConfig(model="ae", z_dim=zz).name for zz in sel["z_grid"]]
    runs += [TrainConfig(model="vae", z_dim=z, beta=bb).name for bb in sel["beta_grid"]]
    labels = [f"AE, $z={zz}$" for zz in sel["z_grid"]] + \
             [f"VAE, $z={z}$, $\\beta={bb:g}$" for bb in sel["beta_grid"]]
    mse = bold_max([R[r]["mse"] for r in runs], f4, minimize=True)
    r2 = bold_max([R[r]["r2"] for r in runs])
    corr = bold_max([R[r]["mean_corr"] for r in runs])
    lines = []
    for i, (r, lab) in enumerate(zip(runs, labels)):
        sel_mark = "$^\\star$" if r in (ae_run, vae_run) else ""
        kl = f"{R[r]['kl']:.2f}" if "kl" in R[r] else "--"
        au = f"{R[r]['active_units']}/{z}" if "active_units" in R[r] else "--"
        lines.append(f"{lab}{sel_mark} & {mse[i]} & {r2[i]} & {corr[i]} & {f4(R[r]['mse_normal'])} & "
                     f"{f4(R[r]['mse_anomaly'])} & {kl} & {au} \\\\")
        if i == len(sel["z_grid"]) - 1:
            lines.append("\\hline")
    with open(GEN_DIR / "tab-recon.tex", "w") as f:
        f.write("\\begin{tabular}{lccccccc}\n\\hline\n"
                "\\textbf{Model} & \\textbf{MSE} & $\\mathbf{R^2}$ & $\\bar{\\boldsymbol{\\rho}}$ & "
                "\\textbf{MSE$_0$} & \\textbf{MSE$_1$} & \\textbf{KL} & \\textbf{AU} \\\\ \\hline\n")
        f.write("\n".join(lines) + "\n\\hline\n\\end{tabular}\n")

    # ---- Table: cumulative anomaly comparison ----------------------------- #
    order = ["Logistic Regression", "Random Forest", "Gradient Boosting", "Supervised MLP (W2 ref.)",
             "AE", "VAE (recon)", "VAE (recon + KL)", "Denoising AE"]
    c = cum.set_index("model").loc[order].reset_index()
    names = {"Supervised MLP (W2 ref.)": "MLP (W2 ref.)", "VAE (recon)": "VAE: recon.",
             "VAE (recon + KL)": "VAE: recon.\\,+\\,KL", "Denoising AE": "Denoising AE",
             "Logistic Regression": "Logistic Reg.", "Random Forest": "Random Forest",
             "Gradient Boosting": "Gradient Boost.", "AE": "AE"}
    sup_mask = c["supervision"] == "supervised"
    rows = []
    unsup_best = {m: c.loc[~sup_mask, m].max() for m in ("roc_auc", "pr_auc", "precision", "recall", "f1")}
    for i, r in c.iterrows():
        cells = []
        for m in ("roc_auc", "pr_auc", "precision", "recall", "f1"):
            v = f3(r[m])
            if not sup_mask[i] and abs(r[m] - unsup_best[m]) < 1e-12:
                v = f"\\textbf{{{v}}}"
            cells.append(v)
        ws = "W1" if r["workshop"] == "W1" else ("W2$^\\dagger$" if sup_mask[i] else "W4")
        rows.append(f"{ws} & {names[r['model']]} & " + " & ".join(cells) + " \\\\")
        if i == 3:
            rows.append("\\hline")
    with open(GEN_DIR / "tab-anomaly.tex", "w") as f:
        f.write("\\begin{tabular}{llccccc}\n\\hline\n"
                "& \\textbf{Model} & \\textbf{ROC} & \\textbf{PR} & \\textbf{P} & \\textbf{R} & "
                "\\textbf{F1} \\\\ \\hline\n")
        f.write("\n".join(rows) + "\n\\hline\n\\end{tabular}\n")

    # ---- Table: sweep (regimes) ------------------------------------------ #
    rows = []
    for zz in sel["z_grid"]:
        n = S[TrainConfig(model="ae", regime="normal", z_dim=zz).name]
        a = S[TrainConfig(model="ae", regime="all", z_dim=zz).name]
        mark = "$^\\star$" if zz == z else ""
        rows.append(f"AE, $z={zz}${mark} & {f3(n['val_auc'])} & {f3(n['roc_auc'])} & {f3(a['roc_auc'])} \\\\")
    rows.append("\\hline")
    for bb in sel["beta_grid"]:
        n = S[TrainConfig(model="vae", regime="normal", z_dim=z, beta=bb).name + " [mse]"]
        nk = S[TrainConfig(model="vae", regime="normal", z_dim=z, beta=bb).name + " [mse+kl]"]
        a = S[TrainConfig(model="vae", regime="all", z_dim=z, beta=bb).name + " [mse]"]
        mark = "$^\\star$" if bb == b else ""
        rows.append(f"VAE, $\\beta={bb:g}${mark} & {f3(n['val_auc'])} & {f3(n['roc_auc'])} "
                    f"({f3(nk['roc_auc'])}) & {f3(a['roc_auc'])} \\\\")
    with open(GEN_DIR / "tab-sweep.tex", "w") as f:
        f.write("\\begin{tabular}{lccc}\n\\hline\n"
                "& \\multicolumn{2}{c}{\\textbf{Normal rows only}} & \\textbf{All rows} \\\\\n"
                "\\textbf{Model} & \\textbf{Val} & \\textbf{Test} & \\textbf{Test} \\\\ \\hline\n")
        f.write("\n".join(rows) + "\n\\hline\n\\end{tabular}\n")

    # ---- Table: latent separability --------------------------------------- #
    reps = ["Input space (16-D)", "AE", "VAE", "Supervised (W2 ref.)"]
    rnames = {"Input space (16-D)": "Input features", "AE": "AE code", "VAE": "VAE $\\mu$",
              "Supervised (W2 ref.)": "MLP (W2 ref.)$^\\dagger$"}
    rows = []
    for r in reps:
        m = L[r]
        tsne = f3(m["silhouette_tsne_2d"]) if "silhouette_tsne_2d" in m else "--"
        umap = f3(m["silhouette_umap_2d"]) if "silhouette_umap_2d" in m else "--"
        rows.append(f"{rnames[r]} & {m['dim']} & {f3(m['silhouette']['full'])} & "
                    f"{f3(m['silhouette']['balanced'])} & {tsne} & {umap} & {f3(m['probe']['roc_auc'])} & "
                    f"{f3(m['probe']['pr_auc'])} \\\\")
    with open(GEN_DIR / "tab-latent.tex", "w") as f:
        f.write("\\begin{tabular}{lccccccc}\n\\hline\n"
                "& & \\multicolumn{4}{c}{\\textbf{Silhouette}} & \\multicolumn{2}{c}{\\textbf{Linear probe}} \\\\\n"
                "\\textbf{Space} & \\textbf{Dim} & \\textbf{All} & \\textbf{Bal.} & \\textbf{t-SNE} & "
                "\\textbf{UMAP} & \\textbf{ROC} & \\textbf{PR} \\\\ \\hline\n")
        f.write("\n".join(rows) + "\n\\hline\n\\end{tabular}\n")

    print(f"LaTeX files written to {GEN_DIR}")
    for k in ("zstar", "betastar", "aeAUC", "vaeklAUC", "supAUC", "silVAE", "silSup"):
        print(f"  \\{k} = {macros[k]}")


if __name__ == "__main__":
    main()

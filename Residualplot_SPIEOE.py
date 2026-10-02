import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


root_dirs = {
    'Berek':          r"...Berek",
    'Soleil-Babinet': r"...Soleil_Babinet",
    'Waveplates':     r"...Waveplates",
    'ref':            r"...Unkompensiert",
}

plotfolder = r"...\images"
os.makedirs(plotfolder, exist_ok=True)


# Mittelwertfunktion
def robust_mean(values, z_thresh=2.5):
    #Robuster Mittelwert
    values = np.asarray(values, dtype=float)
    median = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - median))

    if mad == 0:
        return median

    z = 0.6745 * (values - median) / mad
    filtered = values[np.abs(z) < z_thresh]
    return np.mean(filtered) if len(filtered) else median


num_messreihen = 5
num_messungen  = 7
ordner_namen   = [f"Messreihe{i}" for i in range(1, num_messreihen + 1)]
parameter_namen = ['Normalized s 1', 'Normalized s 2', 'Normalized s 3']
messung_index  = np.arange(1, num_messungen + 1)

ergebnisse = {}


for root_name, root_dir in root_dirs.items():
    data = np.zeros((num_messungen, num_messreihen, 3))

    for messung_idx in range(num_messungen):
        for reihe_idx, ordner in enumerate(ordner_namen):
            datei_ordner = os.path.join(root_dir, ordner)
            dateien = sorted(f for f in os.listdir(datei_ordner) if f.endswith('.csv'))

            if messung_idx >= len(dateien):
                raise IndexError(f"Nicht genug CSV-Dateien in {datei_ordner}")

            datei_pfad = os.path.join(datei_ordner, dateien[messung_idx])

            df = pd.read_csv(
                datei_pfad,
                sep=';', header=7, decimal='.',
                na_values=[' No new data.'],
                skip_blank_lines=True, engine='python',
            )

            werte_df = df.iloc[:, 2:5].apply(pd.to_numeric, errors='coerce')

            werte = np.array([
                robust_mean(werte_df[col].dropna()) for col in werte_df.columns
            ])
            data[messung_idx, reihe_idx, :] = werte

    means = np.mean(data, axis=1)
    stds  = np.std(data, axis=1, ddof=1)
    ergebnisse[root_name] = (means, stds)


# Residuenplot
means_ref, stds_ref = ergebnisse['ref']

colors   = {'Berek': 'tab:blue', 'Soleil-Babinet': 'tab:orange', 'Waveplates': '#2e7d32'}
markers  = {'Berek': 'o', 'Soleil-Babinet': 's', 'Waveplates': '^'}
offsets  = {'Berek': -0.08, 'Soleil-Babinet': 0.0, 'Waveplates': 0.08}

for param_idx in range(3):
    plt.figure(figsize=(10, 6))

    for root_name in ['Berek', 'Soleil-Babinet', 'Waveplates']:
        means, stds = ergebnisse[root_name]
        residuen = means[:, param_idx] - means_ref[:, param_idx]
        err      = 2 * stds[:, param_idx] + 2 * stds_ref[:, param_idx]

        plt.errorbar(
            messung_index + offsets[root_name],
            residuen,
            yerr=err,
            fmt=markers[root_name],
            linestyle='none',
            color=colors[root_name],
            markersize=5,
            elinewidth=1.1,
            capsize=5,
            capthick=1.1,
            alpha=0.9,
            label=root_name,
        )

    plt.axhline(0, color='0.3', linestyle='--', linewidth=1)
    plt.title(
        f"Residuals of different compensation methods relative to reference: "
        f"{parameter_namen[param_idx]}",
        fontsize=14,
    )
    plt.xlabel('Measurement number', fontsize=14)
    plt.ylabel('Residuals', fontsize=14)
    plt.grid(True)
    plt.legend(frameon=False, fontsize=9)
    plt.tick_params(direction='in', top=True, right=True, labelsize=12)
    plt.minorticks_on()
    plt.tick_params(axis='x', which='minor', bottom=False)
    plt.tight_layout()

    filename = os.path.join(plotfolder, f"residuals_stokes_param{param_idx + 1}.pdf")
    plt.savefig(filename, dpi=600, bbox_inches='tight')
    plt.close()

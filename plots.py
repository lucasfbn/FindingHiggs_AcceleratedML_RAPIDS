import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import seaborn as sns


def _palette(x): return sns.color_palette("Blues_d", n_colors=x, desat=0.6)


def class_balance(df):
    plt.clf()
    palette = _palette(2)
    palette.reverse()

    fig, ax = plt.subplots(figsize=(15, 8))
    sns.barplot(x=df["class"].value_counts().index, y=df["class"].value_counts() / len(df) * 100,
                palette=palette)
    plt.xlabel("Classes")
    plt.ylabel("Distribution (relative)")
    plt.title("Relative distribution - class labels")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(20)

    fig.tight_layout()


def boxplot_prep(df, cols, col):
    df = df[cols]
    df = df.melt()
    df["variable"] = df["variable"].str.replace(col, "")
    return df


def boxplots(df, col):
    plt.clf()
    palette = _palette(5)
    palette.reverse()

    fig, ax = plt.subplots(figsize=(15, 8))
    sns.boxplot(x="variable", y="value", data=df, palette=palette)
    plt.xlabel(col)
    plt.ylabel("Value")
    plt.title(f"Distribution of {col} among the jets and the lepton")

    return fig, ax


def boxplot_jets_lepton_wrapper(df):
    combinations = [
        (['jet 1 pt', 'jet 2 pt', 'jet 3 pt', 'jet 4 pt', "lepton pT"], "momentum"),
        (['jet 1 eta', 'jet 2 eta', 'jet 3 eta', 'jet 4 eta', "lepton eta"], "theta"),
        (['jet 1 phi', 'jet 2 phi', 'jet 3 phi', 'jet 4 phi', "lepton phi"], "phi"),
    ]

    for cols, col in combinations:
        plt.clf()
        melted = boxplot_prep(df, cols, col)
        fig, ax = boxplots(melted, col)

        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(20)

        fig.tight_layout()
        plt.show()


def cm(df):
    fig, ax = plt.subplots(figsize=(15, 8))
    df = df.drop(columns=["class"] + ['jet 1 b-tag', 'jet 2 b-tag', 'jet 3 b-tag', 'jet 4 b-tag'])

    df = df.rename(columns={"missing energy magnitude": "M.E. magnitude", "missing energy phi": "M.E. phi"})
    sns.heatmap(df.corr(), annot=False)

    fig.tight_layout()


def btags(df):
    cols = ['jet 1 b-tag', 'jet 2 b-tag', 'jet 3 b-tag', 'jet 4 b-tag']

    vc = []
    for col in cols:
        temp = df[col].value_counts().to_frame(name="count")
        temp["tag"] = temp.index
        temp = temp.sort_values(by="tag")
        temp["tag_idx"] = [0, 1, 2]

        temp["count_roll"] = temp["count"].cumsum()
        temp["jet"] = col.replace("b-tag", "").strip()
        temp = temp.reset_index(drop=True)
        vc.append(temp)

    vc = pd.concat(vc)
    grps = vc.groupby(by="tag_idx", sort=False)

    color_codes = ["pastel", "muted", "dark"]

    fig, ax = plt.subplots(figsize=(15, 8))

    for (name, grp), cc in zip(reversed(tuple(grps)), color_codes):
        sns.set_color_codes(cc)

        label = "b-tag = 0.0" if name == 0 else f"b-tag > {name}.0"

        sns.barplot(x="jet", y="count_roll", data=grp, color="b", label=label)

    ax.legend(ncol=1, loc="upper right", frameon=True, fontsize=20)
    ax.set(ylabel="Total obs. per b-tag bin",
           xlabel="Jets")
    plt.title("Binned distribution of b-tags over jets")

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(20)

    fig.tight_layout()


def missing_boxplot(df):
    cols = ['missing energy magnitude', 'missing energy phi']
    df = boxplot_prep(df, cols, "missing energy")

    plt.clf()
    palette = _palette(2)
    palette.reverse()

    fig, ax = plt.subplots(figsize=(15, 8))
    sns.boxplot(x="variable", y="value", data=df, palette=palette)
    plt.xlabel("Missing energy")
    plt.ylabel("Value")
    plt.title("Distribution of missing energy magnitude/phi")

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(20)

    fig.tight_layout()


def high_level_features_boxplots(df):
    cols = ['m_jj', 'm_jjj', 'm_lv', 'm_jlv', 'm_bb', 'm_wbb', 'm_wwbb']
    df = boxplot_prep(df, cols, "")

    plt.clf()
    palette = _palette(7)
    palette.reverse()

    fig, ax = plt.subplots(figsize=(15, 8))
    g = sns.boxplot(x="variable", y="value", data=df, palette=palette, showfliers=False)
    plt.xlabel("Feature")
    plt.ylabel("Value")
    plt.title("Distributions of high-level features (without outliers)")

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(20)

    fig.tight_layout()

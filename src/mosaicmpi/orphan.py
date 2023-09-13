    # for dataset_name, programs in selected_programs.droplevel(axis=1, level=[2,3]).T.groupby(level=1):
    #     programs = programs.droplevel(level=1).T
    #     programs.columns = programs.columns.astype("int")
    #     for feature in config.features["features_of_interest"]:
    #         if feature in programs.index:
    #             positive_scores = programs.loc[feature].groupby(level=0).mean().reindex(communities.keys()).fillna(0).clip(lower=0)
    #             if positive_scores.sum() == 0:
    #                 continue
    #             fig = plot_community_network(
    #                 graph = Gcomm,
    #                 layout = community_layout,
    #                 plot_size = config.integrate["plot_size_community"],
    #                 title = f"Dataset: {dataset_name}\nFeature: {feature}",
    #                 node_sizes=positive_scores,
    #                 edge_weights="n_edges",
    #                 config=config,
    #                 community_colors=community_colors
    #             )
    #             os.makedirs(os.path.join(output_dir, "annotated_communities", "cNMF_score"), exist_ok=True)
    #             fig.savefig(os.path.join(output_dir, "annotated_communities", "cNMF_score", f"{feature}_{dataset_name}.pdf"))
    #             plt.close(fig)
        
    # # per patient community level plots
    # if any(["patient_id_column" in d for d in config.datasets.values()]):
    #     patient_to_samples = {patient: [] for sample, patient in sample_to_patient.items()}
    #     for sample, patient in sample_to_patient.items():
    #         patient_to_samples[patient].append(sample)
    #     patient_to_samples = pd.Series(patient_to_samples).explode()

    #     n_cols = 4
    #     for annotation_layer in merged_metadata.select_dtypes("category").columns:
    #         if annotation_layer == "Dataset":
    #             colordict = {dsname: dsparam["color"] for dsname, dsparam in config.datasets.items()}
    #         else:
    #             colordict = config.get_metadata_colors(annotation_layer)
    #             colordict[""] = config.metadata_colors["missing_data"]
    #         for min_samples_per_patient in [1,2]:
    #             n_plots = (patient_to_samples.groupby(level=[0,1]).count() >= min_samples_per_patient).sum() + 1  # one plot per patient as well as an extra for the legend.
    #             n_rows = 1 + n_plots // n_cols
    #             fig, axes = plt.subplots(n_rows, n_cols, figsize = [config.integrate["plot_size_community"][0]*n_cols, config.integrate["plot_size_community"][1]*n_rows], squeeze=False, layout="constrained")
    #             for row in range(n_rows):
    #                 for col in range(n_cols):
    #                     axes[row,col].set_axis_off()

    #             # Add legend
    #             ax = axes[0, 0]  # get lower right axes
    #             legend_elements = []
    #             for name, color in colordict.items():
    #                 legend_elements.append(Line2D([0], [0], marker='o', color='w', label=name, markerfacecolor=color, markersize=10))
    #             ax.legend(handles=legend_elements, loc='center')
    #             ax.set_title("Legend", fontdict={'fontsize': 26})

    #             plot_count = 1
    #             for (dataset, patient), patient_samples in patient_to_samples.groupby(level=[0,1]):
    #                 if patient_samples.shape[0] >= min_samples_per_patient:
    #                     ax = axes[plot_count // n_cols, plot_count % n_cols]
    #                     bar_data = ic_usage.loc[patient_samples].fillna(0)
    #                     bar_data.index = merged_metadata.loc[bar_data.index][annotation_layer].cat.add_categories("").fillna("")
    #                     bar_data = bar_data.groupby(level=0).mean().dropna(how="any")
    #                     plot_overrepresentation_network(Gcomm, community_layout, f"{dataset}\n{patient}", overrepresentation=bar_data, colordict=colordict, pie_size=config.integrate["pie_size_community"], ax=ax, edge_weights=None, show_legends=False)
    #                     plot_count += 1
    #             os.makedirs(os.path.join(output_dir, "annotated_communities", "patient_network", annotation_layer), exist_ok=True)
    #             fig.savefig(os.path.join(output_dir, "annotated_communities", "patient_network", annotation_layer, f"{min_samples_per_patient}samplesperpatient.pdf"))
    #             fig.savefig(os.path.join(output_dir, "annotated_communities", "patient_network", annotation_layer, f"{min_samples_per_patient}samplesperpatient.png"), dpi=100)
    #             plt.close("all")
        
    # # Plot pairwise correlation heatmaps of community usage across samples
    # def plot_icusage_correlation(ic_usage_corr, title=None):
    #     mask = np.triu(np.ones_like(ic_usage_corr), 1)
    #     fig, ax = plt.subplots(figsize=[8,6])
    #     sns.heatmap(ic_usage_corr, center=0, vmin=-1, vmax=1, cmap=config.colormaps["diverging"], mask=mask, ax=ax)
    #     ax.set_title(title)
    #     return fig

    # plots = {"All Datasets": plot_icusage_correlation(ic_usage.dropna(axis=1).corr("spearman"), title="All Datasets")}
    # plots.update({
    #     dataset_name: plot_icusage_correlation(df.corr("spearman"), title=dataset_name)
    #     for dataset_name, df in ic_usage.dropna(axis=1).groupby(level=0)
    #     })

    # os.makedirs(os.path.join(output_dir, "integrated_community_usage", "correlation_heatmaps_shared"), exist_ok=True)
    # for plot_name, fig in plots.items():
    #     fig.savefig(os.path.join(output_dir, "integrated_community_usage", "correlation_heatmaps_shared", plot_name + ".pdf"))
    #     fig.savefig(os.path.join(output_dir, "integrated_community_usage", "correlation_heatmaps_shared", plot_name + ".png"), dpi=600)
        
    # plots = {"All Datasets": plot_icusage_correlation(ic_usage.corr("spearman"), title="All Datasets")}
    # plots.update({
    #     dataset_name: plot_icusage_correlation(df.corr("spearman"), title=dataset_name)
    #     for dataset_name, df in ic_usage.groupby(level=0)
    #     })

    # os.makedirs(os.path.join(output_dir, "integrated_community_usage", "correlation_heatmaps_all"), exist_ok=True)
    # for plot_name, fig in plots.items():
    #     fig.savefig(os.path.join(output_dir, "integrated_community_usage", "correlation_heatmaps_all", plot_name + ".pdf"))
    #     fig.savefig(os.path.join(output_dir, "integrated_community_usage", "correlation_heatmaps_all", plot_name + ".png"), dpi=600)
    
    
    # # Diversity analysis (shannon entropy of community-level usage)

    # diversity = ic_usage.apply(lambda x: entropy(x.dropna()), axis=1)
    # diversity.to_csv(os.path.join(output_dir, "integrated_community_usage", "diversity.txt"), sep="\t")
    # os.makedirs(os.path.join(output_dir, "integrated_community_usage", "diversity"), exist_ok=True)

    # for col in merged_metadata.select_dtypes("float").columns:  # association of diversity with numerical metadata
    #     df = pd.DataFrame({"diversity": diversity, col: merged_metadata[col]})
    #     df["Dataset"] = df.index.get_level_values(0)
    #     fig, ax = plt.subplots(figsize=[4,4])
    #     sns.scatterplot(data=df, x=col, y="diversity", hue="Dataset", ax=ax)
    #     correlation = merged_metadata[col].corr(diversity, method="spearman")
    #     ax.text(s=f"Spearman ρ = {correlation:.3f}", x=0.01, y=0.01, va="bottom", transform=ax.transAxes)
    #     ax.set_title(col)
    #     sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    #     fig.savefig(os.path.join(output_dir, "integrated_community_usage", "diversity", f"{col}.pdf"))
    #     plt.close('all')

    # sample_groups = merged_metadata["Dataset"]  # association of diversity with dataset
    # fig = plot_icu_diversity(sample_groups, diversity, config, title=f"Datasets")
    # fig.savefig(os.path.join(output_dir, "integrated_community_usage", "diversity", f"datasets.pdf"), bbox_inches="tight")
    # fig.savefig(os.path.join(output_dir, "integrated_community_usage", "diversity", f"datasets.png"), bbox_inches="tight", dpi=400)
    # plt.close('all')

    # for dataset in config.datasets: # association of diversity with categorical data by dataset
    #     os.makedirs(os.path.join(output_dir, "integrated_community_usage", "diversity", dataset), exist_ok=True)   
    #     for annotation_layer in merged_metadata.select_dtypes("category").columns:
    #         sample_groups = merged_metadata.loc[dataset, annotation_layer]
    #         if 0 < sample_groups.nunique() < 20:
    #             fig = plot_icu_diversity(sample_groups, diversity.loc[dataset], config, title=f"{dataset}\n{annotation_layer}")
    #             fig.savefig(os.path.join(output_dir, "integrated_community_usage", "diversity", dataset, f"{annotation_layer}.pdf"), bbox_inches="tight")
    #             fig.savefig(os.path.join(output_dir, "integrated_community_usage", "diversity", dataset, f"{annotation_layer}.png"), bbox_inches="tight", dpi=400)
    #             plt.close('all')
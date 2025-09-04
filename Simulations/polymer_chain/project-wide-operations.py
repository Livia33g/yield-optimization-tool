import sys
import json
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (helpful on clusters)
import matplotlib.pyplot as plt
from statistics import mean, stdev

def save_polymer_distribution():
    """
    Reads each job’s polymer_dist.json and target_cluster_yield.json,
    aggregates them into polymer_dist_by_kT.json. Additionally, tracks
    the maximum chain size encountered for each concentration.
    """
    import signac

    project = signac.get_project()

    polymer_size_by_kT = {}
    concentrations = {job.sp.concentration for job in project}

    for concentration in concentrations:
        conc_key = f"concentration={concentration}"
        polymer_size_by_kT[conc_key] = {}
        # Track largest chain for this concentration
        polymer_size_by_kT[conc_key]["_largest_chain"] = 1  # Start minimal

        for job in project.find_jobs({"concentration": concentration}):
            # Only include jobs that have target_cluster_yield.json
            if not job.isfile("target_cluster_yield.json"):
                continue

            kT_key = f"kT={job.sp.kT}"
            rep_key = f"replica {job.sp.replica}"

            # Ensure nested dict
            if kT_key not in polymer_size_by_kT[conc_key]:
                polymer_size_by_kT[conc_key][kT_key] = {}

            with job:
                # 1) Read polymer_dist.json
                with open("polymer_dist.json", "r") as dist_file:
                    dist_data = json.load(dist_file)
                avg_counts = dist_data["Averages"]

                # Convert string counts to float
                for size_key in avg_counts:
                    avg_counts[size_key] = float(avg_counts[size_key])

                # Find the largest chain size in this job’s data
                # e.g. size_key is "7 monomers", parse out the integer "7"
                job_largest = max(int(k.split()[0]) for k in avg_counts.keys())
                # Update the concentration’s global largest chain
                if job_largest > polymer_size_by_kT[conc_key]["_largest_chain"]:
                    polymer_size_by_kT[conc_key]["_largest_chain"] = job_largest

                total_count = sum(avg_counts.values())

                # 2) Read target_cluster_yield.json
                with open("target_cluster_yield.json", "r") as abc_file:
                    abc_list = json.load(abc_file)  # e.g. [0.1388...]
                abc_fraction = float(abc_list[0]) if abc_list else 0.0

            # 3) Prepare categories
            category_counts = {
                "1 monomers": 0.0,
                "2 monomers": 0.0,
                "3 monomers": 0.0,  # off-target trimer portion
                "ABC": 0.0,        # from target_cluster_yield
                "4 monomers": 0.0,
                "5 monomers": 0.0,
                "≥6 monomers": 0.0
            }

            # 4) Convert ABC fraction to a count
            abc_count = abc_fraction * total_count

            # 5) Distribute counts
            for size_key, count_val in avg_counts.items():
                size_int = int(size_key.split()[0])
                if size_int < 6:
                    if size_int == 3:
                        # Subtract the ABC portion from the 3-mers
                        leftover_3 = count_val - abc_count
                        if leftover_3 < 0:
                            leftover_3 = 0.0
                        category_counts["3 monomers"] += leftover_3
                        category_counts["ABC"] += abc_count
                    else:
                        cat_label = f"{size_int} monomers"
                        category_counts[cat_label] += count_val
                else:
                    category_counts["≥6 monomers"] += count_val

            polymer_size_by_kT[conc_key][kT_key][rep_key] = category_counts

        # Remove kT keys that ended up empty
        # (i.e., if no job in that kT had the files)
        valid_kTs = {}
        for kT_key, val in polymer_size_by_kT[conc_key].items():
            # Skip if it's just the _largest_chain or if empty
            if kT_key.startswith("_"):
                continue
            if val:
                valid_kTs[kT_key] = val
        polymer_size_by_kT[conc_key] = {
            **valid_kTs,
            "_largest_chain": polymer_size_by_kT[conc_key]["_largest_chain"]
        }

        # ---------------------------------------------------------------------
        # Average & stdev across replicas for each (kT, category)
        # ---------------------------------------------------------------------
        for kT_key in polymer_size_by_kT[conc_key]:
            if kT_key.startswith("_"):
                # skip special keys
                continue

            # Gather all categories
            all_cats = set()
            for rep_key in polymer_size_by_kT[conc_key][kT_key]:
                all_cats.update(polymer_size_by_kT[conc_key][kT_key][rep_key].keys())

            data_per_cat = {cat: [] for cat in all_cats}
            # Collect replicate counts
            for rep_key in polymer_size_by_kT[conc_key][kT_key]:
                rep_dict = polymer_size_by_kT[conc_key][kT_key][rep_key]
                for cat in all_cats:
                    data_per_cat[cat].append(rep_dict.get(cat, 0.0))

            # Compute average & stdev
            Average = {}
            Stdev = {}
            for cat, values in data_per_cat.items():
                avg_val = mean(values)
                Average[cat] = avg_val
                if len(values) > 1:
                    Stdev[cat] = stdev(values)
                else:
                    Stdev[cat] = 0.0

            polymer_size_by_kT[conc_key][kT_key]["Average"] = Average
            polymer_size_by_kT[conc_key][kT_key]["Std_Dev"] = Stdev

        # Sort the kT keys numerically
        sorted_kT = {}
        for kT_key in sorted(valid_kTs.keys(), key=lambda item: float(item.split("=")[1])):
            sorted_kT[kT_key] = polymer_size_by_kT[conc_key][kT_key]
        # Keep the special _largest_chain
        sorted_kT["_largest_chain"] = polymer_size_by_kT[conc_key]["_largest_chain"]
        polymer_size_by_kT[conc_key] = sorted_kT

    # Save final JSON
    with open("polymer_dist_by_kT.json", "w") as out_file:
        json.dump(polymer_size_by_kT, out_file, indent=4)

def plot_polymer_distribution():
    """
    Plots monomer, dimer, off-target trimer, ABC target, and 4–(largest chain).
    The largest chain number is determined from the "_largest_chain" field
    in polymer_dist_by_kT.json for each concentration.

    The figure title is:
    "Cluster yields vs Temperature (Mass action law + stronger loss function)"
    Produces one PNG per concentration.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # Larger fonts, thicker lines, etc.
    plt.rcParams.update({
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "lines.linewidth": 2.5,
        "figure.figsize": (10, 6),
    })

    # Load aggregated data
    with open("polymer_dist_by_kT.json", "r") as f:
        distribution = json.load(f)

    # We'll define categories to sum or plot
    # Format: (plot_label, list_of_original_keys, color, linestyle, marker)
    # We'll rename the last label dynamically below.
    categories_to_plot = [
        ("monomer",           ["1 monomers"],                     "#1f77b4", "-",  "o"),
        ("dimers",             ["2 monomers"],                     "#2ca02c", "--", "s"),
        ("Off-target trimers", ["3 monomers"],                     "#ff7f0e", "-.", None),
        ("ABC target",        ["ABC"],                            "#d62728", ":",  "*"),
        ("≥4 monomers",       ["4 monomers","5 monomers","≥6 monomer chain"], 
                                                          "#9467bd", "-",  "D"),
    ]

    for concentration in distribution.keys():
        # skip the special case if it has no data
        if not distribution[concentration]:
            continue

        # Pull out the largest chain
        # It's stored in distribution[concentration]["_largest_chain"]
        # We'll rename the last category label accordingly.
        largest_chain = 4
        if "_largest_chain" in distribution[concentration]:
            largest_chain = distribution[concentration]["_largest_chain"]

        # Create a copy of categories_to_plot so we can rename the last label
        categories_for_plot = []
        for (lbl, keys, color, ls, marker) in categories_to_plot:
            if lbl == "≥4 monomers":
                if largest_chain <= 4:
                    new_label = "≥4 monomers"
                else:
                    new_label = f"4–{largest_chain} monomers"
                categories_for_plot.append((new_label, keys, color, ls, marker))
            else:
                categories_for_plot.append((lbl, keys, color, ls, marker))

        # We store kT-based data in the aggregator with keys like "kT=0.8"
        # so let's skip the special "_largest_chain"
        kT_keys = [k for k in distribution[concentration] if not k.startswith("_")]
        if not kT_keys:
            continue

        print(f"Plotting for {concentration} ...")

        # We'll collect kT values, fraction, and error for each category
        kT_vals = []
        cat_fractions = {cat[0]: [] for cat in categories_for_plot}
        cat_errors    = {cat[0]: [] for cat in categories_for_plot}

        # Gather data for each kT
        for kT_key in kT_keys:
            kT_val = float(kT_key.split("=")[1])
            kT_vals.append(kT_val)

            avg_counts = distribution[concentration][kT_key]["Average"]
            std_counts = distribution[concentration][kT_key]["Std_Dev"]
            total_count = sum(avg_counts.values())

            for (new_label, old_keys, _, _, _) in categories_for_plot:
                sum_avg = 0.0
                sum_var = 0.0
                for key in old_keys:
                    if key in avg_counts:
                        sum_avg += avg_counts[key]
                        if key in std_counts:
                            sum_var += (std_counts[key])**2

                frac = sum_avg / total_count if total_count > 0 else 0.0
                frac_err = math.sqrt(sum_var) / total_count if total_count > 0 else 0.0

                cat_fractions[new_label].append(frac)
                cat_errors[new_label].append(frac_err)

        # Make the plot
        fig, ax = plt.subplots()

        kT_vals = np.array(kT_vals)
        inv_kT = 1.0 / kT_vals
        # Sort by ascending 1/kT
        sort_idx = np.argsort(inv_kT)

        for (label, _, color, linestyle, marker) in categories_for_plot:
            yvals = np.array(cat_fractions[label])[sort_idx]
            yerrs = np.array(cat_errors[label])[sort_idx]

            ax.errorbar(
                inv_kT[sort_idx],
                yvals,
                yerr=yerrs,
                capsize=4,
                label=label,
                color=color,
                linestyle=linestyle,
                marker=marker,
                markersize=8 if marker else 0,
            )

        ax.set_xlabel("1/kT")
        ax.set_ylabel("yields")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"    Cluster yields vs Temperature (Mass action law) \n{concentration}")
        ax.grid(True, linestyle=":", alpha=0.7)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
        plt.subplots_adjust(right=0.75)

        outname = f"cluster_yields_{concentration}.svg"
        plt.savefig(outname, dpi=300)
        plt.close()
        print(f"Saved {outname}.")

if __name__ == "__main__":
    # Example usage:
    #   python script.py save_polymer_distribution
    #   python script.py plot_polymer_distribution_custom
    if len(sys.argv) > 1:
        func_name = sys.argv[1]
        if func_name in globals():
            globals()[func_name]()
        else:
            print(f"No function named '{func_name}' found.")
    else:
        print("No argument provided. Options: save_polymer_distribution or plot_polymer_distribution_custom.")

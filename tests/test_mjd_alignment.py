"""Test script to verify MJD alignment fix."""
import numpy as np
import matplotlib.pyplot as plt
import antares_client
import pandas as pd

def main():
    """Test the MJD alignment fix."""
    # Get a real light curve
    transient_ztf_id = 'ZTF18abdscvj'
    locus = antares_client.search.get_by_ztf_object_id(ztf_object_id=transient_ztf_id)
    df_ref = locus.timeseries.to_pandas()
    
    # Extract light curve data
    df_ref_g = df_ref[(df_ref.ant_passband == "g") & (~df_ref.ant_mag.isna())]
    df_ref_r = df_ref[(df_ref.ant_passband == "R") & (~df_ref.ant_mag.isna())]
    
    # Create dummy timeseries without MJD values
    timeseries_df = pd.DataFrame({
        'obs_num': range(50),  # Make 50 observation points
        'g_peak_mag': np.random.normal(20, 0.5, 50),
        'r_peak_mag': np.random.normal(19.5, 0.5, 50),
    })
    
    # Add dummy anomaly scores (random for this test)
    anomaly_scores = np.random.normal(-0.2, 0.5, 50)
    pred_prob_anom = np.zeros((len(anomaly_scores), 2))
    for i, score in enumerate(anomaly_scores):
        # Convert decision scores to probability-like values (0-100 scale)
        # Lower scores = more anomalous
        anomaly_prob = 100 * (1 / (1 + np.exp(score)))  # Sigmoid function
        normal_prob = 100 - anomaly_prob
        pred_prob_anom[i, 0] = normal_prob  # normal probability
        pred_prob_anom[i, 1] = anomaly_prob  # anomaly probability
    
    # Calculate the MJD range from the actual light curve
    min_mjd = min(df_ref_g['ant_mjd'].min(), df_ref_r['ant_mjd'].min())
    max_mjd = max(df_ref_g['ant_mjd'].max(), df_ref_r['ant_mjd'].max())
    mjd_margin = (max_mjd - min_mjd) * 0.05
    mjd_range = (min_mjd - mjd_margin, max_mjd + mjd_margin)
    
    # Fix 1: Create a properly scaled mjd_cutoff column
    print("Creating properly scaled mjd_cutoff column...")
    timeseries_df['mjd_cutoff'] = np.linspace(min_mjd, max_mjd, len(timeseries_df))
    
    # Create a plot to demonstrate the fix
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
    
    # Plot the light curve
    ax1.invert_yaxis()
    ax1.errorbar(
        x=df_ref_r.ant_mjd,
        y=df_ref_r.ant_mag,
        yerr=df_ref_r.ant_magerr,
        fmt="o",
        c="r",
        label=r"ZTF-$r$",
    )
    ax1.errorbar(
        x=df_ref_g.ant_mjd,
        y=df_ref_g.ant_mag,
        yerr=df_ref_g.ant_magerr,
        fmt="o",
        c="g",
        label=r"ZTF-$g$",
    )
    
    # Fix 2: Set the x-axis range to match the light curve data
    ax1.set_xlim(mjd_range)
    
    # Fix 3: Only plot the anomaly data points that fall within the light curve's time range
    mask = (timeseries_df.mjd_cutoff >= mjd_range[0]) & (timeseries_df.mjd_cutoff <= mjd_range[1])
    anomaly_mjd = timeseries_df.mjd_cutoff[mask]
    anomaly_prob_normal = pred_prob_anom[mask, 0]
    anomaly_prob_anomaly = pred_prob_anom[mask, 1]
    
    # Plot anomaly probabilities
    ax2.plot(
        anomaly_mjd,
        anomaly_prob_normal,
        drawstyle="steps",
        label=r"$p(Normal)$",
    )
    ax2.plot(
        anomaly_mjd,
        anomaly_prob_anomaly,
        drawstyle="steps",
        label=r"$p(Anomaly)$",
    )
    
    # Add labels and adjust appearance
    ax1.set_title(f"{transient_ztf_id} - MJD Alignment Test")
    ax1.set_ylabel("Magnitude")
    ax2.set_ylabel("Probability (%)")
    ax2.set_xlabel("MJD")
    
    ax1.legend()
    ax2.legend()
    
    ax1.grid(True)
    ax2.grid(True)
    
    # Before the fix, we would see indices 0-50 instead of actual MJD values
    # After the fix, the x-axes should align and use real MJD values
    plt.savefig("mjd_alignment_test.png")
    print("Saved test plot to mjd_alignment_test.png")
    print("Light curve MJD range:", mjd_range)
    print("Anomaly detection MJD values:", min(anomaly_mjd), "to", max(anomaly_mjd))

if __name__ == "__main__":
    main() 
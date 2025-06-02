# Light-curve features for reLAISS - comprehensive set including extreme transient detection
lc_features_const = [
    # Core supernova timing and brightness features
    "g_peak_mag",
    "r_peak_mag", 
    "g_peak_time",
    "r_peak_time",
    "g_rise_time",
    "g_decline_time",
    "r_rise_time",
    "r_decline_time",
    "g_duration_above_half_flux",
    "r_duration_above_half_flux",
    
    # Amplitude and variability features
    "g_amplitude",
    "r_amplitude",
    "g_skewness",
    "r_skewness",
    "g_beyond_2sigma",
    "r_beyond_2sigma",
    
    # Color features
    "mean_g-r",
    "g-r_at_g_peak",
    "mean_color_rate",
    
    # Peak structure features
    "g_n_peaks",
    "r_n_peaks",
    "g_dt_main_to_secondary_peak",
    "r_dt_main_to_secondary_peak",
    "g_dmag_secondary_peak",
    "r_dmag_secondary_peak",
    "g_secondary_peak_prominence", 
    "r_secondary_peak_prominence",
    "g_secondary_peak_width",
    "r_secondary_peak_width",
    
    # Rolling variance features
    "g_max_rolling_variance",
    "r_max_rolling_variance",
    "g_mean_rolling_variance",
    "r_mean_rolling_variance",
    
    # Local curvature features
    "g_rise_local_curvature",
    "g_decline_local_curvature",
    "r_rise_local_curvature",
    "r_decline_local_curvature",
    # "total_duration", # useful but requires re-calculating on the full feature bank
]

# Host feature list for LAISS. You can comment out features you want to exclude.
host_features_const = [
    "gKronMagCorrected",
    "gKronRad",
    "gExtNSigma",
    "rKronMagCorrected",
    "rKronRad",
    "rExtNSigma",
    "iKronMagCorrected",
    "iKronRad",
    "iExtNSigma",
    "zKronMagCorrected",
    "zKronRad",
    "zExtNSigma",
    "gminusrKronMag",
    "rminusiKronMag",
    "iminuszKronMag",
    "rmomentXX",
    "rmomentXY",
    "rmomentYY",
]

############# DO NOT CHANGE CONSTANTS BELOW THIS LINE #############

lc_feature_err = [
    "g_peak_mag_err",
    "r_peak_mag_err",
    "g_peak_time_err",
    "g_rise_time_err",
    "g_decline_time_err",
    "g_duration_above_half_flux_err",
    "r_duration_above_half_flux_err",
    "r_peak_time_err",
    "r_rise_time_err",
    "r_decline_time_err",
    "mean_g-r_err",
    "g-r_at_g_peak_err",
    "mean_color_rate_err",
    "g_mean_rolling_variance_err",
    "r_mean_rolling_variance_err",
    "g_rise_local_curvature_err",
    "g_decline_local_curvature_err",
    "r_rise_local_curvature_err",
    "r_decline_local_curvature_err",
]

host_feature_err = [
    "gKronMagErr",
    "rKronMagErr",
    "iKronMagErr",
    "gminusrKronMagErr",
    "rminusiKronMagErr",
    "iminuszKronMagErr",
]


err_lookup = {
    # Lightcurve feature error names
    "g_peak_mag": "g_peak_mag_err",
    "r_peak_mag": "r_peak_mag_err",
    "g_peak_time": "g_peak_time_err",
    "g_rise_time": "g_rise_time_err",
    "g_decline_time": "g_decline_time_err",
    "g_duration_above_half_flux": "g_duration_above_half_flux_err",
    "r_duration_above_half_flux": "r_duration_above_half_flux_err",
    "r_peak_time": "r_peak_time_err",
    "r_rise_time": "r_rise_time_err",
    "r_decline_time": "r_decline_time_err",
    "mean_g-r": "mean_g-r_err",
    "g-r_at_g_peak": "g-r_at_g_peak_err",
    "mean_color_rate": "mean_color_rate_err",
    "g_mean_rolling_variance": "g_mean_rolling_variance_err",
    "r_mean_rolling_variance": "r_mean_rolling_variance_err",
    "g_rise_local_curvature": "g_rise_local_curvature_err",
    "g_decline_local_curvature": "g_decline_local_curvature_err",
    "r_rise_local_curvature": "r_rise_local_curvature_err",
    "r_decline_local_curvature": "r_decline_local_curvature_err",
    # Host feature error names
    "gKronMagCorrected": "gKronMagErr",
    "rKronMagCorrected": "rKronMagErr",
    "iKronMagCorrected": "iKronMagErr",
    "gminusrKronMag": "gminusrKronMagErr",
    "rminusiKronMag": "rminusiKronMagErr",
    "iminuszKronMag": "iminuszKronMagErr",
}


# All features from dataset bank needed to engineer host features
raw_host_features_const = [
    "gKronMag",
    "gKronMagErr",
    "gKronRad",
    "gExtNSigma",
    "rmomentXX",
    "rmomentYY",
    "rmomentXY",
    "rKronMag",
    "rKronMagErr",
    "rKronRad",
    "rExtNSigma",
    "iKronMag",
    "iKronMagErr",
    "iKronRad",
    "iExtNSigma",
    "zKronMag",
    "zKronMagErr",
    "zKronRad",
    "zExtNSigma",
]

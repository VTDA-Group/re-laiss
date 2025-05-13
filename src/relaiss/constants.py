# Lightcurve feature list for LAISS. You can comment out features you want to exclude.
lc_features_const = [
    "g_peak_mag",
    "r_peak_mag",
    "g_peak_time",
    "g_rise_time",
    "g_decline_time",
    "g_duration_above_half_flux",
    "r_duration_above_half_flux",
    "r_peak_time",
    "r_rise_time",
    "r_decline_time",
    "mean_g-r",
    "g-r_at_g_peak",
    "mean_color_rate",
    "g_mean_rolling_variance",
    "r_mean_rolling_variance",
    "g_rise_local_curvature",
    "g_decline_local_curvature",
    "r_rise_local_curvature",
    "r_decline_local_curvature",
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

import geopandas
import pandas
import shapely
import numpy as np
from tqdm import tqdm


def main():
    wgi = pandas.read_csv("wgi_feb2012.csv", encoding="ISO-8859-1")
    rgi = geopandas.read_file("rgi.shp").to_crs("epsg:4326")
    aoi = geopandas.read_file("aoi.shp").to_crs("epsg:4326").geometry.iloc[0]

    wgi = wgi[wgi.total_area > 2]
    wgi = wgi[wgi.apply(lambda x: shapely.Point(x["lon"], x["lat"]).within(aoi), axis=1)]
    wgi["year"] = wgi.photo_year
    # 24 glaciers < 8 km2 are missing year information, we impute them in a dirty way
    # given that they account for 0.24% of the area and that the imputing error is ~1 year
    # we can confidently claim that it does not affect the eventual trend reestimation
    wgi = impute_missing_years(wgi)

    rgi = rgi[rgi.intersects(aoi)]
    rgi = rgi[rgi.area_km2 > 2]
    rgi["year"] = rgi.src_date.map(lambda x: int(x[:4]))

    wgi_area_km2 = wgi.total_area.sum()
    rgi_area_km2 = rgi.area_km2.sum()
    
    uncertainty_range_km2 = 500
    area_uncertainty_mult_sigma = uncertainty_range_km2 / 33775 / 1.96 # from Nuth et al., 2013
    n = 100000
    # n = 1000 # use for debugging

    wgi_composition = [
        (year, wgi[wgi.year == year].total_area.sum() / wgi_area_km2)
        for year in wgi.year.unique()
    ]
    rgi_composition = [
        (year, rgi[rgi.year == year].area_km2.sum() / rgi_area_km2)
        for year in rgi.year.unique()
    ]

    wgi_x_mean = np.average([_[0] for _ in wgi_composition], weights=[_[1] for _ in wgi_composition])
    rgi_x_mean = np.average([_[0] for _ in rgi_composition], weights=[_[1] for _ in rgi_composition])
    # print(wgi_x_mean, rgi_x_mean)
    
    # for s, y_ref, a in [(wgi_composition, wgi_x_mean, wgi_area_km2), (rgi_composition, rgi_x_mean, rgi_area_km2)]:
    #     sum = 0
    #     for y, w in s:
    #         delta_f = 0.0023 if y < y_ref else -0.0023
    #         sum += (delta_f * w * (y - y_ref))
    #     print(sum)
    #     sum *= a
    #     print(f"delta_f sensitivity bias at 0.0023 = {sum} km2")
    # return => 0.97% conservative error in area => sigma ~ 0.005 (rounding up)
    # uncertainty_mult_sigma = np.sqrt(uncertainty_mult_sigma**2 + 0.005**2)

    # for s in [wgi_composition, rgi_composition]:
    #     for y, w in s:
    #         print(y, w  * 100)
    #     print()
    # 
    # print(wgi_area_km2, area_uncertainty_mult_sigma * wgi_area_km2)
    # print(rgi_area_km2, area_uncertainty_mult_sigma * rgi_area_km2)
    # print()
    
    # Stage 1, initial guess
    trends_frac, trends_km2 = mc_run(
        wgi_area_km2, rgi_area_km2, 
        area_uncertainty_mult_sigma,
        wgi_composition, rgi_composition, 
        0,
        n,
    )

    print(f"Trend ({wgi_x_mean:.2f}--{rgi_x_mean:.2f}) statistics (initial guess):")
    mean_f_initial = np.mean(trends_frac)
    print(f"\t {mean_f_initial:.6f} +- {1.96 * np.std(trends_frac):.6f} \t 95%-CI:[{np.percentile(trends_frac, 2.5):.6f};{np.percentile(trends_frac, 97.5):.6f}] \t Median:{np.median(trends_frac):.6f} \t p.a.")
    print(f"\t {np.mean(trends_km2):.3f} +- {1.96 * np.std(trends_km2):.3f} \t 95%-CI:[{np.percentile(trends_km2, 2.5):.3f};{np.percentile(trends_km2, 97.5):.3f}] \t Median:{np.median(trends_km2):.3f} \t km2 p.a.")
    print()

    # Stage 2, pessimistic w.r.t. delta_f variance
    trends_frac, trends_km2 = mc_run(
        wgi_area_km2, rgi_area_km2, 
        area_uncertainty_mult_sigma,
        wgi_composition, rgi_composition, 
        np.abs(mean_f_initial),
        n,
    )

    print(f"Trend ({wgi_x_mean:.2f}--{rgi_x_mean:.2f}) statistics (corrected for delta_f):")
    print(f"\t {np.mean(trends_frac):.6f} +- {1.96 * np.std(trends_frac):.6f} \t 95%-CI:[{np.percentile(trends_frac, 2.5):.6f};{np.percentile(trends_frac, 97.5):.6f}] \t Median:{np.median(trends_frac):.6f} \t p.a.")
    print(f"\t {np.mean(trends_km2):.3f} +- {1.96 * np.std(trends_km2):.3f} \t 95%-CI:[{np.percentile(trends_km2, 2.5):.3f};{np.percentile(trends_km2, 97.5):.3f}] \t Median:{np.median(trends_km2):.3f} \t km2 p.a.")
    print()
    
    # compare with ours
    our_trend_km2_pa = -260.22
    our_trend_conservative_km2_pa = -227.62727999996935
    our_stddev = 74.57134931380898 / 1.96 
    # ours passed several normality tests, so sample from normal
    # factors = np.random.normal(loc=our_trend_km2_pa, scale=our_stddev, size=n) / \
    #     np.random.choice(trends_km2, size=n, replace=True)
    factors_cons = np.random.normal(loc=our_trend_conservative_km2_pa, scale=our_stddev, size=n) / \
        np.random.choice(trends_km2, size=n, replace=True)
    print("Acceleration factor:")
    # print(f"\t x{np.mean(factors):.3f} +- {1.96 * np.std(factors):.3f} \t 95%-CI:[{np.percentile(factors, 2.5):.3f};{np.percentile(factors, 97.5):.3f}] \t (with >2km2 in ours)")
    print(f"\t x{np.mean(factors_cons):.3f} +- {1.96 * np.std(factors_cons):.3f} \t 95%-CI:[{np.percentile(factors_cons, 2.5):.3f};{np.percentile(factors_cons, 97.5):.3f}] \t Median:{np.median(factors_cons):.3f} \t (without >2km2 in ours; uncertainty conservative)")


def mc_run(
    wgi_area_km2, rgi_area_km2, 
    area_uncertainty_mult_sigma,
    wgi_composition, rgi_composition, 
    delta_f_sigma,
    n,
): 
    k_eff1 = 1 / np.sum([_[1]**2 for _ in wgi_composition])
    k_eff2 = 1 / np.sum([_[1]**2 for _ in rgi_composition])
    
    trends_km2 = []
    trends_frac = []
    k_eff1_running, k_eff2_running = 0, 0

    for _ in tqdm(range(n), total=n):
        size1 = int(k_eff1_running + k_eff1 - int(k_eff1_running)) # correction for non-integer k_eff
        size2 = int(k_eff2_running + k_eff2 - int(k_eff2_running))
        x1 = np.random.choice(
            [_[0] for _ in wgi_composition], 
            p=[_[1] for _ in wgi_composition],
            size=size1, replace=True,
        )
        x2 = np.random.choice(
            [_[0] for _ in rgi_composition], 
            p=[_[1] for _ in rgi_composition],
            size=size2, replace=True,
        )
        x1_mean = np.mean(x1)
        x2_mean = np.mean(x2)
        
        y1 = wgi_area_km2 * np.random.normal(loc=1, scale=area_uncertainty_mult_sigma)
        y2 = rgi_area_km2 * np.random.normal(loc=1, scale=area_uncertainty_mult_sigma)
        
        if delta_f_sigma > 0:
            x1_, w1_ = np.unique(x1, return_counts=True)
            x2_, w2_ = np.unique(x2, return_counts=True)
            delta_f1 = np.random.normal(loc=0, scale=delta_f_sigma, size=len(w1_))
            delta_f2 = np.random.normal(loc=0, scale=delta_f_sigma, size=len(w2_))
            corr1 = np.sum(delta_f1 * w1_ * (x1_mean - x1_)) / np.sum(w1_)
            corr2 = np.sum(delta_f2 * w2_ * (x2_mean - x2_)) / np.sum(w2_)
            y1 *= (1 + corr1)
            y2 *= (1 + corr2)
        
        t = (y2 - y1) / (x2_mean - x1_mean)
        f = t / y1
        trends_km2.append(t)
        trends_frac.append(f)
        k_eff1_running += k_eff1
        k_eff2_running += k_eff2
    
    return trends_frac, trends_km2
        
    
def impute_missing_years(df):
    from sklearn.model_selection import GridSearchCV, KFold
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.neighbors import KNeighborsRegressor

    # Split into known and unknown year subsets
    df_known = df[df["year"].notna()].copy()
    df_unknown = df[df["year"].isna()].copy()
    # close (basin, lon, lat) glaciers are more likely to be mapped from the same imagery
    # glaciers from the same data source are more likely to be mapped from the same imagery
    X_known = df_known[["drainage_code", "lat", "lon", "data_contributor"]] 
    y_known = df_known["year"].astype(int)

    # Build the pipeline
    categorical_cols = ["drainage_code", "data_contributor"]
    numeric_cols = ["lat", "lon"]

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", Pipeline([("scaler", StandardScaler())]), numeric_cols),
        ],
        remainder="drop", 
    )
    knn = KNeighborsRegressor()
    pipe = Pipeline(
        steps=[
            ("prep", preprocess),
            ("knn", knn),
        ]
    )

    # Cross-validation to select the best model, probably an overkill here but let's follow best practices
    param_grid = {
        "knn__n_neighbors": [1, 3, 5, 7, 9],
        "knn__weights": ["uniform", "distance"],
    }
    cv = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="neg_mean_absolute_error",
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=-1,
    )

    cv.fit(X_known, y_known)
    best_model = cv.best_estimator_

    # Predict the missing years
    X_unknown = df_unknown[["drainage_code", "lat", "lon", "data_contributor"]]
    year_pred = best_model.predict(X_unknown)
    df_unknown["year"] = year_pred.round().astype(int)
    df_imputed = pandas.concat([df_known, df_unknown], ignore_index=True)
    return df_imputed


if __name__ == "__main__":
    np.random.seed(42) # fixed to reproduce the exact numbers from the manuscript
    main()

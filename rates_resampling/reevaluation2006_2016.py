import geopandas
import pandas
import shapely
import numpy as np
from tqdm import tqdm


def main():
    rgi = geopandas.read_file("rgi.shp").to_crs("epsg:4326")
    aoi = geopandas.read_file("aoi.shp").to_crs("epsg:4326").geometry.iloc[0]
    rgi = rgi[rgi.intersects(aoi)]
    rgi = rgi[rgi.area_km2 > 2]
    rgi["year"] = rgi.src_date.map(lambda x: int(x[:4]))
    rgi_area_km2 = rgi.area_km2.sum()
    
    uncertainty_range_km2 = 500
    area_uncertainty_mult_sigma = uncertainty_range_km2 / 33775 / 1.96 # from Nuth et al., 2013
    n = 100000
    n = 1000 # use for debugging

    rgi_composition = [
        (year, rgi[rgi.year == year].area_km2.sum() / rgi_area_km2)
        for year in rgi.year.unique()
    ]
    rgi_x_mean = np.average([_[0] for _ in rgi_composition], weights=[_[1] for _ in rgi_composition])
    
    bootstrap_2016 = pandas.read_csv("2016bootstrap.csv")["bootstrapped areas in 2016, km2 "].values
    correction_2016 = np.mean(bootstrap_2016) - 31889.2797 # align the bootstrapping distribution with >2km2 threshold
    bootstrap_2016 = bootstrap_2016 - correction_2016
    x2 = 2016

    # Stage 1, initial guess
    trends_frac, trends_km2 = mc_run(
        rgi_area_km2, 
        area_uncertainty_mult_sigma,
        rgi_composition, 
        x2, bootstrap_2016,
        0,
        n,
    )

    print(f"Trend ({rgi_x_mean:.2f}--{x2}) statistics (initial guess):")
    mean_f_initial = np.mean(trends_frac)
    print(f"\t {mean_f_initial:.6f} +- {1.96 * np.std(trends_frac):.6f} \t 95%-CI:[{np.percentile(trends_frac, 2.5):.6f};{np.percentile(trends_frac, 97.5):.6f}] \t Median:{np.median(trends_frac):.6f} \t p.a.")
    print(f"\t {np.mean(trends_km2):.3f} +- {1.96 * np.std(trends_km2):.3f} \t 95%-CI:[{np.percentile(trends_km2, 2.5):.3f};{np.percentile(trends_km2, 97.5):.3f}] \t Median:{np.median(trends_km2):.3f} \t km2 p.a.")
    print()
    
    # Stage 2, pessimistic w.r.t. delta_f variance
    trends_frac, trends_km2 = mc_run(
        rgi_area_km2, 
        area_uncertainty_mult_sigma,
        rgi_composition, 
        x2, bootstrap_2016,
        np.abs(mean_f_initial),
        n,
    )

    print(f"Trend ({rgi_x_mean:.2f}--{x2}) statistics (corrected for delta_f):")
    print(f"\t {np.mean(trends_frac):.6f} +- {1.96 * np.std(trends_frac):.6f} \t 95%-CI:[{np.percentile(trends_frac, 2.5):.6f};{np.percentile(trends_frac, 97.5):.6f}] \t Median:{np.median(trends_frac):.6f} \t p.a.")
    print(f"\t {np.mean(trends_km2):.3f} +- {1.96 * np.std(trends_km2):.3f} \t 95%-CI:[{np.percentile(trends_km2, 2.5):.3f};{np.percentile(trends_km2, 97.5):.3f}] \t Median:{np.median(trends_km2):.3f} \t km2 p.a.")
    print()


def mc_run(
    rgi_area_km2, 
    area_uncertainty_mult_sigma,
    rgi_composition, 
    x2, bootstrap_areas,
    delta_f_sigma,
    n,
): 
    k_eff = 1 / np.sum([_[1]**2 for _ in rgi_composition])
    
    trends_km2 = []
    trends_frac = []
    k_eff_running = 0

    for _ in tqdm(range(n), total=n):
        size = int(k_eff_running + k_eff - int(k_eff_running)) # correction for non-integer k_eff
        
        x1 = np.random.choice(
            [_[0] for _ in rgi_composition], 
            p=[_[1] for _ in rgi_composition],
            size=size, replace=True,
        )
        x1_mean = np.mean(x1)
        
        y1 = rgi_area_km2 * np.random.normal(loc=1, scale=area_uncertainty_mult_sigma)
        y2 = np.random.choice(bootstrap_areas)
        
        if delta_f_sigma > 0:
            x1_, w1_ = np.unique(x1, return_counts=True)
            delta_f1 = np.random.normal(loc=0, scale=delta_f_sigma, size=len(w1_))
            corr1 = np.sum(delta_f1 * w1_ * (x1_mean - x1_)) / np.sum(w1_)
            y1 *= (1 + corr1)
        
        t = (y2 - y1) / (x2 - x1_mean)
        f = t / y1
        trends_km2.append(t)
        trends_frac.append(f)
        k_eff_running += k_eff
    
    return trends_frac, trends_km2


if __name__ == "__main__":
    np.random.seed(42) # fixed to reproduce the exact numbers from the manuscript
    main()

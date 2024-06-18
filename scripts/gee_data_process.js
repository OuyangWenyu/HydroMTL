function calculateNldasAvg(year_num, region_name, region_shape) {
    var year = ee.Number(year_num);
    var month = ee.Number(1);
    var day = ee.Number(1);
    var start_date = ee.Date.fromYMD(year, month, day);
    // var end_date = start_date.advance(1, 'year');
    var end_date = start_date.advance(1, 'day');
    print(end_date);
    var days_num = end_date.difference(start_date, 'day');
    print("number of days", days_num);
    // count day from zero, and ee.List.sequence is a closed interval
    var days = ee.List.sequence(ee.Number(0), days_num.add(-1));
    print(days);

    // get Imagecollection and filter, choose two days for test
    var nldas_2d = nldas.filter(ee.Filter.date(start_date, end_date));
    print(nldas_2d.limit(10));

    // forcing variables that will be calculated for its avg
    var avg_forcings = nldas_2d.select('temperature', 'specific_humidity', 'pressure', 'wind_u', 'wind_v', 'longwave_radiation', 'convective_fraction', 'shortwave_radiation');
    print(avg_forcings.limit(10));

    // forcing variables that will be calculated for its sum
    var sum_forcings = nldas_2d.select('potential_energy', 'potential_evaporation', 'total_precipitation');
    print(sum_forcings.limit(10));

    // show temperature, just for test
    var temperature = nldas_2d.select('temperature');
    Map.addLayer(temperature, temperatureVis, 'Temperature');
    print(temperature.limit(10));


    // avg every day
    var avg_days = ee.ImageCollection(days.map(function (day) {
        var start = start_date.advance(ee.Number(day), 'day');
        var end = start_date.advance(ee.Number(day).add(ee.Number(1)), 'day');
        return avg_forcings.filter(ee.Filter.date(start, end)).reduce(ee.Reducer.mean()).set({
            'day_of_all_years': start_date.advance(day, 'day')
        });
    }));
    print(avg_days.limit(10));

    // sum every day
    var sum_days = ee.ImageCollection(days.map(function (day) {
        var start = start_date.advance(ee.Number(day), 'day');
        var end = start_date.advance(ee.Number(day).add(ee.Number(1)), 'day');
        return sum_forcings.filter(ee.Filter.date(start, end)).reduce(ee.Reducer.sum()).set({
            'day_of_all_years': start_date.advance(day, 'day')
        });
    }));
    print(sum_days.limit(10));

    // show avg temperature of all days, just for test
    var tmpr_avg = avg_days.select('temperature_mean').reduce(ee.Reducer.mean());
    print(tmpr_avg);
    var temperature_avg = tmpr_avg.select('temperature_mean_mean');
    Map.addLayer(temperature_avg, temperatureVis, 'Temperature_2d_avg');

    //map and reduce to get the mean of day-mean forcings of every regions in everyday
    var avg_day_regions = avg_days.map(function (img) {
        return img.reduceRegions({
            collection: region_shape,
            reducer: ee.Reducer.mean(),
            scale: 13875
        }).map(function (feature) {
            return feature.set({
                "time_start": img.get("day_of_all_years"),
                "gage_id": feature.get("hru_id"),
            });
        });
    }).flatten();
    print("avg_day_regions", avg_day_regions.limit(5));

    //map and reduce to get the mean of day-sum forcings of every regions in everyday
    var avg_day_regions_4sum = sum_days.map(function (img) {
        return img.reduceRegions({
            collection: region_shape,
            reducer: ee.Reducer.mean(),
            scale: 13875
        }).map(function (feature) {
            return feature.set({
                "time_start": img.get("day_of_all_years"),
                "gage_id": feature.get("hru_id"),
            });
        });
    }).flatten();
    print("avg_day_regions_4sum", avg_day_regions_4sum.limit(5));


    // export avg_day_regions to google drive
    Export.table.toDrive({
        collection: avg_day_regions,   // The feature collection to export.
        description: "NLDAS_" + region_name + "_mean_" + year_num,   // A human-readable name of the task. Defaults to "myExportTableTask".
        folder: 'NLDAS',  // The Google Drive Folder that the export will reside in.
        fileNamePrefix: "NLDAS_" + region_name + "_mean_" + year_num,  // The filename prefix. Defaults to the task's description.
        selectors: ["gage_id", "time_start", "temperature_mean", "specific_humidity_mean", "pressure_mean", "wind_u_mean", "wind_v_mean", "longwave_radiation_mean", "convective_fraction_mean", "shortwave_radiation_mean"]  // A list of properties to include in the export; either a single string with comma-separated names or a list of strings.
    });
    // export avg_day_regions_4sum to google drive
    Export.table.toDrive({
        collection: avg_day_regions_4sum,   // The feature collection to export.
        description: "NLDAS_" + region_name + "_sum_" + year_num,   // A human-readable name of the task. Defaults to "myExportTableTask".
        folder: 'NLDAS',  // The Google Drive Folder that the export will reside in.
        fileNamePrefix: "NLDAS_" + region_name + "_sum_" + year_num,  // The filename prefix. Defaults to the task's description.
        selectors: ["gage_id", "time_start", "potential_energy_sum", "potential_evaporation_sum", "total_precipitation_sum"]  // A list of properties to include in the export; either a single string with comma-separated names or a list of strings.
    });
}

// var nldas = ee.ImageCollection("NASA/NLDAS/FORA0125_H002"),
// temperatureVis = {"min":-5,"max":40,"palette":["3d2bd8","4e86da","62c7d8","91ed90","e4f178","ed6a4c"]},
// camels = ee.FeatureCollection("users/wenyu_ouyang/Camels/HCDN_nhru_final_671");
// var year_start = 2001;
// var year_end = 2002;
// var year_num;
// var str_camels = "camels";
// var str_camels591 = "camels591";
// for(year_num=year_start;year_num<year_end;year_num++){
//   calculateAvg(year_num,str_camels,camels);
// calculateAvg(year_num,str_camels591,camels591);
// }


function calculateSmapAvg(year_num, region_name, region_shape, basin_id_name) {
    var year = ee.Number(year_num);
    var month = ee.Number(1);
    var day = ee.Number(1);
    var start_date = ee.Date.fromYMD(year, month, day);
    // var end_date = start_date.advance(1, 'year');
    var end_date = start_date.advance(2, 'day');
    print(end_date);
    var days_num = end_date.difference(start_date, 'day');
    print("number of days", days_num);
    // count day from zero, and ee.List.sequence is a closed interval
    var days = ee.List.sequence(ee.Number(0), days_num.add(-1));
    print(days);

    // get Imagecollection and filter, choose two days for test
    var smap_days = smos.filter(ee.Filter.date(start_date, end_date));
    print(smap_days.limit(10));

    print(region_name, region_shape.limit(5));
    var smap_regions = smap_days.map(function (img) {
        return img.reduceRegions({
            collection: region_shape,
            reducer: ee.Reducer.mean(),
            scale: 10000
        }).map(function (feature) {
            return feature.set({
                "time_start": img.date(),
                "gage_id": feature.get(basin_id_name),
            });
        });
    }).flatten();
    print("smap_regions", smap_regions.limit(5));

    //export to google drive
    Export.table.toDrive({
        collection: smap_regions,   // The feature collection to export.
        description: "smap_" + region_name + "_mean_" + year_num,   // A human-readable name of the task. Defaults to "myExportTableTask".
        folder: 'SMAP',  // The Google Drive Folder that the export will reside in.
        fileNamePrefix: "smap_" + region_name + "_mean_" + year_num,  // The filename prefix. Defaults to the task's description.
        selectors: ["gage_id", "time_start", "ssm", "susm", "smp", "ssma", "susma"]  // A list of properties to include in the export; either a single string with comma-separated names or a list of strings.
    });
}

function calculateEtAvg(year_num, region_name, region_shape, basin_id_name) {
    var year = ee.Number(year_num);
    var month = ee.Number(1);
    var day = ee.Number(1);
    var start_date = ee.Date.fromYMD(year, month, day);
    var end_date = start_date.advance(1, 'year');
    print(end_date);
    var days_num = end_date.difference(start_date, 'day');
    print("number of days", days_num);
    // count day from zero, and ee.List.sequence is a closed interval
    var days = ee.List.sequence(ee.Number(0), days_num.add(-1));
    print(days);

    // get Imagecollection and filter, choose two days for test
    var et_days = mod16a2.filter(ee.Filter.date(start_date, end_date));
    print(et_days.limit(10));

    print(region_name, region_shape.limit(5));
    //map and reduce to get the mean of daily forcings of every regions everyday
    // reduceRegion-version, you can modify it to reduceRegions-version like the above
    var mod16a2_regions = region_shape.map(function (feature) {
        return et_days.filterBounds(feature.geometry()).map(function (img) {
            var vals = img.reduceRegion({
                reducer: ee.Reducer.mean(),
                geometry: feature.geometry(),
                scale: 500
            });
            return ee.Feature(null, vals).set({
                "time_start": img.date(),
                "gage_id": feature.get(basin_id_name)
            });
        });
    }).flatten();
    print("mod16a2_regions", mod16a2_regions.limit(5));

    //export to google drive
    Export.table.toDrive({
        collection: mod16a2_regions,   // The feature collection to export.
        description: "mod16a2006_" + region_name + "_mean_" + year_num,   // A human-readable name of the task. Defaults to "myExportTableTask".
        folder: 'MODIS',  // The Google Drive Folder that the export will reside in.
        fileNamePrefix: "mod16a2006_" + region_name + "_mean_" + year_num,  // The filename prefix. Defaults to the task's description.
        selectors: ["gage_id", "time_start", "ET", "LE", "PET", "PLE", "ET_QC"]  // A list of properties to include in the export; either a single string with comma-separated names or a list of strings.
    });
}
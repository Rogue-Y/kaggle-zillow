feature_list = {
    'before_fill': [
        ('missing_value_count', 'missing_value_count', {}, 'missing_value_count_pickle', False),
    ],
    'original': [
        # required columns
        'parcelid',
        # optional columns
        'latitude',
        'longitude',
        'propertycountylandusecode',
        'propertylandusetypeid',
        'propertyzoningdesc',
        'regionidcity',
        'regionidcounty',
        'regionidneighborhood',
        'regionidzip',
        'fips_census_1',
        'block_1',
        'fips_census_block',
    ],
    'generated': [
        ('geo_city', 'geo_city', {}, 'geo_city_pickle', False),
        ('geo_county', 'geo_county', {}, 'geo_county_pickle', False),
        ('geo_lat_lon_block_features', 'geo_lat_lon_block_features', {}, 'geo_lat_lon_block_features_pickle', False),
        ('geo_fips_census_1', 'geo_fips_census_1', {}, 'geo_fips_census_1_pickle', False),
        ('geo_fips_census_block', 'geo_fips_census_block', {}, 'geo_fips_census_block_pickle', False),
        ('geo_neighborhood', 'geo_neighborhood', {}, 'geo_neighborhood_pickle', False),
        ('geo_zip', 'geo_zip', {}, 'geo_zip_pickle', False),

        ('target_neighborhood_feature', 'target_region_feature', {'id_name':'regionidneighborhood'}, 'target_neighborhood_feature_pickle', False),
        ('target_zip_feature', 'target_region_feature', {'id_name':'regionidzip'}, 'target_zip_feature_pickle', False),
        ('target_city_feature', 'target_region_feature', {'id_name':'regionidcity'}, 'target_city_feature_pickle', False),
        ('target_county_feature', 'target_region_feature', {'id_name':'regionidcounty'}, 'target_county_feature_pickle', False),
        ('target_census_feature', 'target_region_feature', {'id_name':'fips_census_1'}, 'target_census_feature_pickle', False),
        ('target_censusblock_feature', 'target_region_feature', {'id_name':'fips_census_block'}, 'target_censusblock_feature_pickle', False),
    ]
}

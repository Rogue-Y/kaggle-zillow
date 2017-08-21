from .feature_eng import *

feature_list = [
    # ('average_room_size', average_room_size, {}, 'average_room_size_pickle'),
    # ('boolean_has_ac', boolean_has_ac, {}, 'boolean_has_ac_pickle'),
    # ('boolean_has_garage_pool_or_ac', boolean_has_garage_pool_or_ac, {}, 'boolean_has_garage_pool_or_ac_pickle'),
    # ('boolean_has_heat', boolean_has_heat, {}, 'boolean_has_heat_pickle'),
    # ('building_age', building_age, {}, 'building_age_pickle'),
    # ('built_before_year', built_before_year, {}, 'built_before_year_pickle'),
    ('category_ac_type_encode', category_ac_type_encode, {}, 'category_ac_type_encode_pickle'),
    ('category_fips_type_encode', category_fips_type_encode, {}, 'category_fips_type_encode_pickle'),
    ('category_heating_type_encode', category_heating_type_encode, {}, 'category_heating_type_encode_pickle'),
    ('category_land_use_code_encode', category_land_use_code_encode, {}, 'category_land_use_code_encode_pickle'),
    ('category_land_use_desc_encode', category_land_use_desc_encode, {}, 'category_land_use_desc_encode_pickle'),
    ('category_land_use_type_encode', category_land_use_type_encode, {}, 'category_land_use_type_encode_pickle'),
    # ('deviation_from_avg_structure_tax_value', deviation_from_avg_structure_tax_value, {}, 'deviation_from_avg_structure_tax_value_pickle'),
    # ('error_rate_calculated_finished_living_sqft', error_rate_calculated_finished_living_sqft, {}, 'error_rate_calculated_finished_living_sqft_pickle'),
    # ('extra_rooms', extra_rooms, {}, 'extra_rooms_pickle'),
    # ('extra_space', extra_space, {}, 'extra_space_pickle'),
    # ('geo_city_structure_tax_value', geo_city_structure_tax_value, {}, 'geo_city_structure_tax_value_pickle'),
    # ('geo_city_tax_value', geo_city_tax_value, {}, 'geo_city_tax_value_pickle'),
    # ('geo_lat_lon_block', geo_lat_lon_block, {}, 'geo_lat_lon_block_pickle'),
    # ('geo_lat_lon_block_tax_value', geo_lat_lon_block_tax_value, {}, 'geo_lat_lon_block_tax_value_pickle'),
    # ('geo_neighorhood_tax_value', geo_neighorhood_tax_value, {}, 'geo_neighorhood_tax_value_pickle'),
    # ('geo_region_tax_value', geo_region_tax_value, {}, 'geo_region_tax_value_pickle'),
    # ('geo_zip_tax_value', geo_zip_tax_value, {}, 'geo_zip_tax_value_pickle'),
    # ('missing_value_count', missing_value_count, {}, 'missing_value_count_pickle'),
    # ('multiply_lat_lon', multiply_lat_lon, {}, 'multiply_lat_lon_pickle'),
    # ('poly_2_structure_tax_value', poly_2_structure_tax_value, {}, 'poly_2_structure_tax_value_pickle'),
    # ('poly_3_structure_tax_value', poly_3_structure_tax_value, {}, 'poly_3_structure_tax_value_pickle'),
    # ('ratio_living_area', ratio_living_area, {}, 'ratio_living_area_pickle'),
    # ('ratio_structure_tax_value_to_land_tax_value', ratio_structure_tax_value_to_land_tax_value, {}, 'ratio_structure_tax_value_to_land_tax_value_pickle'),
    # ('ratio_tax', ratio_tax, {}, 'ratio_tax_pickle'),
    # ('round_lat', round_lat, {}, 'round_lat_pickle'),
    # ('round_lon', round_lon, {}, 'round_lon_pickle'),
    # ('round_multiply_lat_lon', round_multiply_lat_lon, {}, 'round_multiply_lat_lon_pickle'),
    # ('sum_lat_lon', sum_lat_lon, {}, 'sum_lat_lon_pickle'),
    # ('total_rooms', total_rooms, {}, 'total_rooms_pickle')
]

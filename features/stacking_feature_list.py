feature_list_ridge1 = {
    'original': [
        # required columns
        'parcelid',
        # optional columns
        'bathroomcnt',
        'bedroomcnt',
        'calculatedfinishedsquarefeet',
        'fips',
        'fullbathcnt',
        'lotsizesquarefeet',
        'roomcnt',
        'yearbuilt',
        'structuretaxvaluedollarcnt',
        'taxvaluedollarcnt',
        'assessmentyear',
        'landtaxvaluedollarcnt',
        'taxamount',
    ],
}

feature_list_ridge3 = {
    'original': [
        # required columns
        'parcelid',
        # optional columns
        'bathroomcnt',
        'bedroomcnt',
        'calculatedfinishedsquarefeet',
        'fips',
        # 'fullbathcnt',
        'lotsizesquarefeet',
        # 'roomcnt',
        'yearbuilt',
        'structuretaxvaluedollarcnt',
        'taxvaluedollarcnt',
        # 'assessmentyear',
        # 'landtaxvaluedollarcnt',
        # 'taxamount',
        'latitude',
        'longitude',
    ],
    # 'generated': [
    #     ('average_bedroom_size', 'average_bedroom_size', {}, 'average_bedroom_size_pickle', False),
    #     ('extra_space', 'extra_space', {}, 'extra_space_pickle', False),
    #     ('ratio_living_area', 'ratio_living_area', {}, 'ratio_living_area_pickle', False),
    #     ('ratio_tax', 'ratio_tax', {}, 'ratio_tax_pickle', False),
    #     ('ratio_tax_value_to_structure_value', 'ratio_tax_value_to_structure_value', {}, 'ratio_tax_value_to_structure_value_pickle', False),
    # ]
}

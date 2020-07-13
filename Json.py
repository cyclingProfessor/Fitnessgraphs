import json
import re
import numpy

# read and check file
def get_data(data_file):
    with open(data_file, 'r') as f:
        biology = json.load(f)
    domain = biology['mutations']
    size = numpy.prod([len(c) for c in domain.values()])

    instance = biology['data']
    point = instance['genotypes']
    value = instance['phenotypes']
    error = instance['stdeviations']
    if len(point) != size or len(value) != size or len(error) != size:
        print("There should be the same number of genotypes, phenotypes and standard deviations and it should be the product of the domain sizes")
        return None

    mapper = []
    for locus in range(len(domain)):
        nstr = str(locus)
        if nstr not in domain:
            print("Each gene site must have a set of alleles")
            return None
        mapper.insert(locus, dict((value, index) for index, value in enumerate(domain[nstr])))

    for index in range(len(point)):
        alleles = list(point[index])
        if len(domain) != len(alleles):
            print("Each genotype must have one value for each biallele site")
            return None
        try:
            point[index] = tuple(mapper[coord][c] for coord, c in enumerate(alleles))
        except KeyError:
            print("Each allele must be a possible mutation")
            return None
    return (size, tuple(len(c) for c in domain), list(zip(point, value, error)))



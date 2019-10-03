#! usr/bin/env mysql

# create and build db
CREATE DATABASE iedb_public;
USE iedb_public;
SOURCE /Users/weeder/Data/Proteasome/iedb_public.sql;

# index for faster merge downstream
CREATE INDEX t_ind ON mhc_elution(as_type_id);
CREATE INDEX ta_ind ON mhc_elution(mhc_allele_restriction_id);
CREATE INDEX te_ind ON mhc_elution(curated_epitope_id);
CREATE INDEX to_ind ON mhc_elution(h_organism_id);
CREATE INDEX a_ind ON assay_type(as_type_id);
CREATE INDEX m_ind ON mhc_allele_restriction(mhc_allele_restriction_id);
CREATE INDEX ce_ind ON curated_epitope(curated_epitope_id);
CREATE INDEX o_ind ON object(object_id);
CREATE INDEX e_ind ON curated_epitope(e_object_id);
CREATE INDEX eoo_ind ON epitope_object(object_id);
CREATE INDEX eo_ind ON epitope_object(epitope_id);

# select required columns from each table and join. Filter for human, naturally processed, class I epitopes
CREATE TABLE full_epitope_output SELECT
	t.reference_id, t.curated_epitope_id, t.as_type_id, t.h_organism_id, t.mhc_allele_name, t.ant_type, t.ant_object_id,
	a.category, a.assay_type, a.response,
	m.organism, m.class,
	e.epitope_id, e.linear_peptide_seq, e.linear_peptide_modified_seq, e.linear_peptide_modification,
	o.starting_position, o.ending_position,
	s.accession, s.database, s.sequence,
	art.pubmed_id,
	ha.obi_id
FROM mhc_elution AS t
	INNER JOIN assay_type AS a
		ON t.as_type_id=a.assay_type_id
	INNER JOIN mhc_allele_restriction AS m
		ON t.mhc_allele_restriction_id=m.mhc_allele_restriction_id
	INNER JOIN curated_epitope AS ce
		ON t.curated_epitope_id=ce.curated_epitope_id
	INNER JOIN object AS o
		ON o.object_id=ce.e_object_id
	INNER JOIN epitope_object AS eo
		ON eo.object_id=o.object_id
	INNER JOIN epitope AS e
		ON eo.epitope_id=e.epitope_id
	INNER JOIN organism AS org
		ON t.h_organism_id=org.organism_id
	INNER JOIN source AS s
		ON o.mol2_source_id=s.source_id
	INNER JOIN article AS art
		ON art.reference_id=t.reference_id
	INNER JOIN organism_finder_host_ancestry as ha
		ON o.organism2_id=ha.child_org_id
	WHERE a.category = "Naturally Processed" AND m.class = "I" AND ha.obi_id = "http://purl.obolibrary.org/obo/NCBITaxon_40674";

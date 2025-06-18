import pandas as pd

clinical_df = pd.read_csv("clinical-data/clinical.tsv", sep="\t")
follow_up_df = pd.read_csv("clinical-data/follow_up.tsv", sep="\t")
exposure_df = pd.read_csv("clinical-data/exposure.tsv", sep="\t")
family_history_df = pd.read_csv("clinical-data/family_history.tsv", sep="\t")
pathology_detail_df = pd.read_csv("clinical-data/pathology_detail.tsv", sep="\t")

key = "cases.submitter_id"

follow_up_df['follow_ups.year_of_follow_up'] = pd.to_numeric(
    follow_up_df['follow_ups.year_of_follow_up'], errors='coerce'
)
follow_up_latest = (
    follow_up_df
    .sort_values(by=[key, 'follow_ups.year_of_follow_up'], ascending=[True, False])
    .drop_duplicates(subset=key, keep='first')
)

columns_to_drop = ['project.project_id', 'cases.case_id']

family_history_df = family_history_df.drop(columns=[col for col in columns_to_drop if col in family_history_df.columns])
pathology_detail_df = pathology_detail_df.drop(columns=[col for col in columns_to_drop if col in pathology_detail_df.columns])



merged = clinical_df.merge(follow_up_latest, on=key, how='left') \
                    .merge(exposure_df, on=key, how='left') \
                    .merge(family_history_df, on=key, how='left') \
                    .merge(pathology_detail_df, on=key, how='left')


columns_to_keep = [
    key,
    'diagnoses.primary_diagnosis',
    'diagnoses.morphology',
    'diagnoses.age_at_diagnosis',
    'demographic.gender',
    'demographic.race',
    'diagnoses.progression_or_recurrence',
    'diagnoses.last_known_disease_status',
    'demographic.days_to_death',
    'demographic.vital_status',
    'follow_ups.progression_or_recurrence',
    'follow_ups.days_to_recurrence',
    'follow_ups.disease_response',
    'exposures.tobacco_smoking_status',
    'family_histories.relationship_primary_diagnosis',
    'pathology_details.lymph_node_involvement',
    'pathology_details.residual_tumor'
]


clean_df = merged[columns_to_keep].dropna(how='all', subset=columns_to_keep[1:])


clean_df['demographic.days_to_death'] = pd.to_numeric(clean_df['demographic.days_to_death'], errors='coerce')

clean_df['label'] = (
    (clean_df['diagnoses.progression_or_recurrence'] == 'Yes') |
    (clean_df['follow_ups.progression_or_recurrence'] == 'Yes') |
    (clean_df['demographic.days_to_death'] <= 730)
).fillna(False).astype(int)


clean_df.to_csv("clinical-data/cleaned_leukemia_dataset.csv", index=False)
print("Saved cleaned_leukemia_dataset.csv with shape:", clean_df.shape)

df_test = pd.read_csv(os.path.join(TEST_CSV_PATH_20,'test.csv'))

test_df = clean_dataframe(df,df2,df_test)

test_dataset = ScancerDataset(test_df,transforms=transforms_val)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=8
)
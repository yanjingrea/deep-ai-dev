## to generate weekly report
### generate results
* run the following 3 file to get images
* results will be save to dev directory
* these 3 tiles is independent
1. run `DeepAI_weekly_report/test/scr_ranker_test.py` to get unit ranking u curve images
2. run `DeepAI_weekly_report/test/scr_condo_test.py` to get condo demand curve images
3. run `DeepAI_weekly_report/test/scr_ec_test.py` to get executive condo (ec) demand curve images

### compile weekly report
4. run `DeepAI_weekly_report/scr_write_file.py` to wirte the Latex Code for weekly report compiling
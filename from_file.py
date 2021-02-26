from modules.hd_module import hd_module

train_file = './data/game_dat_long.out'
test_file = './data/game_dat.out'

#train_file = './data/sample_dist_short.out'
#$test_file = './data/sample_dist_short.out'

a = hd_module()
a.train_from_file(train_file)
print(a.hd_program_vec)
print(a.hd_cond_vec)
print(a.num_cond)
a.test_from_file(test_file)

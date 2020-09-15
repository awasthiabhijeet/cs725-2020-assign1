import numpy as np

from LR import analytical_solution, compute_gradients, update_weights

def is_close(v1,v2, allowed_error=0.0001):
	if np.sum((v1-v2)**2) < allowed_error:
		return True
	else:
		return False


def test_case_1():
	features = np.eye(3)
	targets = np.array([5.0,13.0,2.0]).reshape(3,1)

	weights = analytical_solution(feature_matrix=features, targets=targets, C=0.0)

	if np.all(weights==targets):
		print('\nyour analytical solution passed this test case!\n')
	else:
		print('\noops! your analytical solution is failed this test case!\n')

	gradients_1 = compute_gradients(feature_matrix=features, weights=weights, targets=targets, C=0.0)
	#print(gradients_1)
	check_gradients_1 = np.all(gradients_1==0)

	gradients_2 = compute_gradients(feature_matrix=features, weights=np.zeros([3,1]), targets=targets, C=0.0)
	#print(gradients_2)
	true_gradients_2 = -2/3 * targets
	check_gradients_2 = is_close(true_gradients_2,gradients_2)
	#print(check_gradients_1)
	#print(check_gradients_2)

	if check_gradients_1 and check_gradients_2:
		print('\nyour compute_gradients solution passed this test case!\n')
	else:
		print('\noops! your compute_gradients solution is failed this test case!\n')

	updated_weights = update_weights(weights=np.zeros([3,1]), gradients=gradients_2, lr=1)
	check_updated_weights = is_close(-updated_weights, gradients_2)

	if check_updated_weights:
		print('\nyour update_weights solution passed this test case!\n')
	else:
		print('\noops! your update_weights solution is failed this test case!\n')




if __name__ == '__main__':
	test_case_1()
	
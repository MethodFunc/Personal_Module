import numpy as np


class ErrorRateBase:
    def __init__(self, y_true, y_pred):
        self.y_true = np.array(y_true) if not isinstance(y_true, np.ndarray) else y_true
        self.y_pred = np.array(y_pred) if not isinstance(y_pred, np.ndarray) else y_pred
        
        if self.y_true.ndim > 1:
            self.y_true = self.y_true.ravel()
        if self.y_pred.ndim > 1:
            self.y_pred = self.y_pred.ravel()

    @classmethod
    def error_print(cls, error_rate):
        print('=====================================')
        print(f'{cls.__name__} evaluation values')
        print(f'Values: {np.mean(error_rate):.4f}%')
        print('=====================================\n')


class NMAE(ErrorRateBase):
    """
    정규 평균 절대 오차입니다.
    오차율을 계산하여 출력됩니다. 해당 값은 변수에 저장되지 않으며 그저 보여주기만 합니다.
    capacity: 발전기의 최대설비용량을 입력해주세요.
    사용법:
        nmae_score = NMAE(y_true, y_pred)
        nmae_score(capacity)
        
    출력:
       1. 모든 데이터 오차율 계산
       2. 최대설비용량 10% 미만 제외한 오차율 계산
    """

    def __call__(self, capacity: int):
        under_indices = np.where(self.y_true >= capacity * 0.1)
        error_rate = np.abs(np.subtract(self.y_pred, self.y_true)) / capacity * 100
        under_rate = np.abs(np.subtract(self.y_pred[under_indices], self.y_true[under_indices])) / capacity * 100
        print('모든 데이터')
        self.error_print(error_rate)
        print('10% 이하 제거')
        self.error_print(under_rate)


class MAPE(ErrorRateBase):
    """
    오차율을 계산하여 출력됩니다. 해당 값은 변수에 저장되지 않으며 그저 보여주기만 합니다.
    사용법:
        mape_score = MAPE(y_true, y_pred)
        mape_score()
        
    출력:
       1. 모든 데이터 오차율 계산
    """

    def __call__(self):
        error_rate = np.abs(np.subtract(self.y_pred, self.y_true)) / self.y_true * 100
        self.error_print(error_rate)


class MSE(ErrorRateBase):
    """
    오차율을 계산하여 출력됩니다. 해당 값은 변수에 저장되지 않으며 그저 보여주기만 합니다.
    사용법:
        mse_score = MSE(y_true, y_pred)
        mse_score()
        
    출력:
       1. 모든 데이터 오차율 계산
    """

    def __call__(self):
        error_rate = np.power(np.subtract(self.y_pred, self.y_true), 2)
        self.error_print(error_rate)


class MAE(ErrorRateBase):
    """
    오차율을 계산하여 출력됩니다. 해당 값은 변수에 저장되지 않으며 그저 보여주기만 합니다.
    사용법:
        mae_score = MAE(y_true, y_pred)
        mae_score()
        
    출력:
       1. 모든 데이터 오차율 계산
    """

    def __call__(self):
        error_rate = np.abs(np.subtract(self.y_pred, self.y_true))
        self.error_print(error_rate)


class MAPE(ErrorRateBase):
    """
    오차율을 계산하여 출력됩니다. 해당 값은 변수에 저장되지 않으며 그저 보여주기만 합니다.
    사용법:
        mape_score = MAPE(y_true, y_pred)
        mape_score()
        
    출력:
       1. 모든 데이터 오차율 계산
    """

    def __call__(self):
        error_rate = np.abs(np.subtract(self.y_pred, self.y_true)) / self.y_true * 100
        self.error_print(error_rate)


class RMSE(ErrorRateBase):
    """
    오차율을 계산하여 출력됩니다. 해당 값은 변수에 저장되지 않으며 그저 보여주기만 합니다.
    사용법:
        rmse_score = RMSE(y_true, y_pred)
        rmse_score()
        
    출력:
       1. 모든 데이터 오차율 계산
    """

    def __call__(self):
        error_rate = np.sqrt(np.power(np.subtract(self.y_pred, self.y_true), 2))
        self.error_print(error_rate)

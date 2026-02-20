import importlib
import pytorch_lightning as pl
from torch import Tensor
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint

# noinspection PyUnresolvedReferences
from typing import (
    Any,
    Generator,
    Iterator,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Type,
    TypedDict,
    Union,
    runtime_checkable,
)

# noinspection PyUnresolvedReferences
if importlib.util.find_spec("ipywidgets") is not None:
    from tqdm.auto import tqdm as _tqdm
else:
    from tqdm import tqdm as _tqdm

STEP_OUTPUT = Optional[Union[Tensor, Mapping[str, Any]]]


# 进度条功能重写
class CustomProgressBar(TQDMProgressBar):

    def __init__(self, args, max_batches=None):
        super().__init__()
        self.args = args
        self.max_epoch_str_len = len(str(self.args.max_epochs))  # 根据max_epochs确定最大占位符长度
        self.ncols_len = 120

        # 根据训练验证和测试集batch数量确定最大字符占位长度 优化进度条显示
        if max_batches is not None and isinstance(max_batches, int):
            self.max_barch_num_str_len = len(str(max_batches))
            a = "{l_bar}{bar}| {n_fmt:>" + str(self.max_barch_num_str_len)
            b = "}/{total_fmt:>" + str(self.max_barch_num_str_len)
            c = "} [{elapsed}<{remaining}{postfix}]"
            self.bar_formate_ = a + b + c
        else:
            self.max_barch_num_str_len = None
            self.bar_formate_ = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]"

    def get_metrics(self, trainer, model):
        items = super().get_metrics(trainer, model)
        if trainer.training:
            loss = trainer.logged_metrics.get('train_loss_avg_step')
            items['loss'] = f"{loss.item():05.5f}"
            loss_noi = trainer.logged_metrics.get('train_loss_for_xf_avg_step')
            if loss_noi is not None:
                items['loss_xf'] = f"{loss_noi.item():05.5f}"
        if trainer.validating:
            loss = trainer.logged_metrics.get('valid_loss_avg_step')
            items['loss'] = f"{loss.item():05.5f}"
            loss_noi = trainer.logged_metrics.get('valid_loss_for_xf_avg_step')
            if loss_noi is not None:
                items['loss_xf'] = f"{loss_noi.item():05.5f}"
        if trainer.testing:
            loss = trainer.logged_metrics.get('test_loss_avg')
            items['loss'] = f"{loss.item():05.5f}"
        return items

    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.dynamic_ncols = False
        bar.ncols = self.ncols_len
        bar.colour = 'green'

        # 如果未传入max_batches参数则按照dataloader属性确定
        if self.total_train_batches is not None and self.max_barch_num_str_len is None:
            self.max_barch_num_str_len = max(self.total_train_batches, self.total_val_batches)
            self.max_barch_num_str_len = len(str(self.max_barch_num_str_len))
            a = "{l_bar}{bar}| {n_fmt:>" + str(self.max_barch_num_str_len)
            b = "}/{total_fmt:>" + str(self.max_barch_num_str_len)
            c = "} [{elapsed}<{remaining}{postfix}]"
            self.bar_formate_ = a + b + c
        bar.bar_format = self.bar_formate_
        return bar

    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.set_description('    Valid ')
        bar.dynamic_ncols = False
        bar.ncols = self.ncols_len
        bar.bar_format = self.bar_formate_
        return bar

    def init_test_tqdm(self):
        bar = super().init_test_tqdm()
        bar.set_description('    Test  ')
        bar.dynamic_ncols = False
        bar.ncols = self.ncols_len
        bar.bar_format = self.bar_formate_
        return bar

    def on_train_epoch_start(self, trainer: "pl.Trainer", *_: Any) -> None:
        self.train_progress_bar.reset(self.total_train_batches)
        self.train_progress_bar.initial = 0

        description = f"Train {trainer.current_epoch+1:>{self.max_epoch_str_len}}/{self.args.max_epochs:>{self.max_epoch_str_len}}"
        self.train_progress_bar.set_description(description)

    def on_train_batch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        n = batch_idx + 1
        if self._should_update(n, self.train_progress_bar.total):
            _update_n(self.train_progress_bar, n)
            self.train_progress_bar.set_postfix(self.get_metrics(trainer, pl_module))

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not self.train_progress_bar.disable:
            self.train_progress_bar.set_postfix(self.get_metrics(trainer, pl_module))

    def on_validation_batch_start(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            batch: Any,
            batch_idx: int,
            dataloader_idx: int = 0,
    ) -> None:
        if not self.has_dataloader_changed(dataloader_idx):
            return
        self.val_progress_bar = self.init_validation_tqdm()
        self.val_progress_bar.colour = 'red'
        self.val_progress_bar.reset(self.total_val_batches_current_dataloader)
        self.val_progress_bar.initial = 0

        description = f"Valid {trainer.current_epoch+1:>{self.max_epoch_str_len}}/{self.args.max_epochs:>{self.max_epoch_str_len}}"
        self.val_progress_bar.set_description(description)

    def on_validation_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs: STEP_OUTPUT,
            batch: Any,
            batch_idx: int,
            dataloader_idx: int = 0,
    ) -> None:
        n = batch_idx + 1
        if self._should_update(n, self.val_progress_bar.total):
            _update_n(self.val_progress_bar, n)
            self.val_progress_bar.set_postfix(self.get_metrics(trainer, pl_module))

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # self.val_progress_bar.close()
        # if self._train_progress_bar is not None and trainer.state.fn == "fit":
        #     self.train_progress_bar.set_postfix(self.get_metrics(trainer, pl_module))
        #     self.val_progress_bar.set_postfix(self.get_metrics(trainer, pl_module))

        # 保存最后一次验证进度条 注意需要先关闭训练进度条 否则两个进度条位置会调换
        if trainer.current_epoch == self.args.max_epochs - 1:
            if self.train_progress_bar is not None:
                # self.train_progress_bar.set_postfix(self.get_metrics(trainer, pl_module))
                self.train_progress_bar.close()  # 最后关闭训练进度条
            self.val_progress_bar.leave = True
            self.val_progress_bar.close()
        self.reset_dataloader_idx_tracker()

    def on_test_batch_start(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            batch: Any,
            batch_idx: int,
            dataloader_idx: int = 0,
    ) -> None:
        if not self.has_dataloader_changed(dataloader_idx):
            return

        self.test_progress_bar.colour = 'blue'
        self.test_progress_bar.reset(self.total_test_batches_current_dataloader)
        self.test_progress_bar.initial = 0

        description = f"Test  {trainer.current_epoch:>{self.max_epoch_str_len}}/{self.args.max_epochs:>{self.max_epoch_str_len}}"
        self.test_progress_bar.set_description(description)

    def on_test_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs: STEP_OUTPUT,
            batch: Any,
            batch_idx: int,
            dataloader_idx: int = 0,
    ) -> None:
        n = batch_idx + 1
        if self._should_update(n, self.test_progress_bar.total):
            _update_n(self.test_progress_bar, n)
            self.test_progress_bar.set_postfix(self.get_metrics(trainer, pl_module))

    def on_sanity_check_start(self, *_: Any) -> None:
        print('Sanity_check')

    def on_sanity_check_end(self, *_: Any) -> None:
        pass

    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.val_progress_bar = self.init_validation_tqdm()


def Model_checkpoint():
    checkpoint_callback = ModelCheckpoint(monitor='valid_loss_epoch',
                                          filename='valid_monitor-{epoch:04d}-{valid_loss:.2f}',
                                          mode='min',
                                          save_last=True)
    return checkpoint_callback


def _update_n(bar: _tqdm, value: int) -> None:
    if not bar.disable:
        bar.n = value
        bar.refresh()

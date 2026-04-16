import sys
import os
import time
import datetime
import ctypes
import numpy as np
import pyqtgraph as pg
import pyqtgraph.exporters
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QGroupBox, QLabel, QComboBox,
                             QPushButton, QSlider, QCheckBox, QFormLayout,
                             QTabWidget, QSizeGrip, QSizePolicy, QSpinBox,
                             QDialog, QFileDialog, QLineEdit, QMessageBox, QRadioButton, QDoubleSpinBox,
                             QTableWidget, QTableWidgetItem, QHeaderView, QInputDialog, QStackedWidget)
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QImage, QPixmap, QPainter, QPen

# ==========================================
# CTypes 直连底层 DLL
# ==========================================
HAS_MVSDK = False
mvdll = None


class tSdkCameraDevInfo(ctypes.Structure):
    _fields_ = [
        ("acProductSeries", ctypes.c_char * 32),
        ("acProductName", ctypes.c_char * 32),
        ("acFriendlyName", ctypes.c_char * 32),
        ("acLinkName", ctypes.c_char * 32),
        ("acDriverVersion", ctypes.c_char * 32),
        ("acSensorType", ctypes.c_char * 32),
        ("acPortType", ctypes.c_char * 32),
        ("acSn", ctypes.c_char * 32),
        ("uInstance", ctypes.c_uint)
    ]


class tSdkFrameHead(ctypes.Structure):
    _fields_ = [
        ("uiMediaType", ctypes.c_uint),
        ("uBytes", ctypes.c_uint),
        ("iWidth", ctypes.c_int),
        ("iHeight", ctypes.c_int),
        ("iWidthZoomSw", ctypes.c_int),
        ("iHeightZoomSw", ctypes.c_int),
        ("bIsTrigger", ctypes.c_int),
        ("uiTimeStamp", ctypes.c_uint),
        ("uiExpTime", ctypes.c_uint),
        ("fAnalogGain", ctypes.c_float),
        ("iGamma", ctypes.c_int),
        ("iContrast", ctypes.c_int),
        ("iSaturation", ctypes.c_int),
        ("fRgain", ctypes.c_float),
        ("fGgain", ctypes.c_float),
        ("fBgain", ctypes.c_float)
    ]


def init_mvsdk_dll():
    global HAS_MVSDK, mvdll
    import platform
    is_64bit = platform.architecture()[0] == '64bit'
    dll_name = "MVCAMSDK_X64.dll" if is_64bit else "MVCAMSDK.dll"

    search_paths = [os.path.dirname(__file__)]
    search_paths.extend(sys.path)

    expanded_paths = []
    for sp in search_paths:
        if not sp: continue
        expanded_paths.append(sp)
        expanded_paths.append(os.path.join(sp, 'SDK'))
        expanded_paths.append(os.path.join(sp, 'SDK', 'X64' if is_64bit else 'x86'))
        expanded_paths.append(os.path.join(sp, 'SDK', 'x64' if is_64bit else 'X86'))

    for p in expanded_paths:
        full_path = os.path.join(p, dll_name)
        if os.path.exists(full_path):
            try:
                if hasattr(os, 'add_dll_directory'):
                    os.add_dll_directory(p)
                mvdll = ctypes.windll.LoadLibrary(full_path)
                HAS_MVSDK = True

                mvdll.CameraEnumerateDevice.argtypes = [ctypes.POINTER(tSdkCameraDevInfo), ctypes.POINTER(ctypes.c_int)]
                mvdll.CameraInit.argtypes = [ctypes.POINTER(tSdkCameraDevInfo), ctypes.c_int, ctypes.c_int,
                                             ctypes.POINTER(ctypes.c_int)]
                mvdll.CameraPlay.argtypes = [ctypes.c_int]
                mvdll.CameraGetImageBuffer.argtypes = [ctypes.c_int, ctypes.POINTER(tSdkFrameHead),
                                                       ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte)), ctypes.c_uint]
                mvdll.CameraImageProcess.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_ubyte),
                                                     ctypes.POINTER(ctypes.c_ubyte), ctypes.POINTER(tSdkFrameHead)]
                mvdll.CameraReleaseImageBuffer.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_ubyte)]
                mvdll.CameraUnInit.argtypes = [ctypes.c_int]
                mvdll.CameraSetTriggerMode.argtypes = [ctypes.c_int, ctypes.c_int]
                mvdll.CameraSetAeState.argtypes = [ctypes.c_int, ctypes.c_int]
                mvdll.CameraSetExposureTime.argtypes = [ctypes.c_int, ctypes.c_double]
                mvdll.CameraSetAnalogGain.argtypes = [ctypes.c_int, ctypes.c_int]
                mvdll.CameraSetIspOutFormat.argtypes = [ctypes.c_int, ctypes.c_uint]
                break
            except Exception as e:
                pass


init_mvsdk_dll()

try:
    from scipy.special import eval_hermite, genlaguerre

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ==========================================
# 异步取流子线程
# ==========================================
class CameraAcquisitionThread(QThread):
    error_occurred = pyqtSignal(str)

    def __init__(self, main_app, camera_data):
        super().__init__()
        self.main_app = main_app
        self.h_camera = camera_data['hCamera']
        self.buf_size = camera_data['buffer_size']
        self.running = False

        self.out_buffer = ctypes.create_string_buffer(self.buf_size)
        self.p_out = ctypes.cast(self.out_buffer, ctypes.POINTER(ctypes.c_ubyte))
        self.frame_head = tSdkFrameHead()
        self.p_raw = ctypes.POINTER(ctypes.c_ubyte)()

    def run(self):
        self.running = True
        frame_count = 0
        last_fps_time = time.time()

        while self.running:
            try:
                res = mvdll.CameraGetImageBuffer(self.h_camera, ctypes.byref(self.frame_head), ctypes.byref(self.p_raw),
                                                 200)
                if res != 0: continue

                mvdll.CameraImageProcess(self.h_camera, self.p_raw, self.p_out, ctypes.byref(self.frame_head))

                w = self.frame_head.iWidth
                h = self.frame_head.iHeight
                frame_bytes = self.frame_head.uBytes

                if frame_bytes == w * h:
                    frame = np.frombuffer(self.out_buffer, dtype=np.uint8, count=w * h).reshape((h, w))
                elif frame_bytes == w * h * 3:
                    frame = np.frombuffer(self.out_buffer, dtype=np.uint8, count=w * h * 3).reshape((h, w, 3))
                    frame = np.dot(frame[..., :3], [0.1140, 0.5870, 0.2989]).astype(np.uint8)
                else:
                    frame = np.frombuffer(self.out_buffer, dtype=np.uint8, count=w * h).reshape((h, w))

                mvdll.CameraReleaseImageBuffer(self.h_camera, self.p_raw)

                if self.running:
                    self.main_app.latest_frame = frame.copy()
                    self.main_app.current_actual_exp = self.frame_head.uiExpTime / 1000.0

                    frame_count += 1
                    now = time.time()
                    if now - last_fps_time >= 0.5:
                        self.main_app.current_fps = frame_count / (now - last_fps_time)
                        frame_count = 0
                        last_fps_time = now
            except Exception as e:
                if self.running:
                    self.error_occurred.emit(f"相机底层抓流异常: {str(e)}")

    def stop(self):
        self.running = False
        self.wait()


# ==========================================
# ★ 核心多线程：独立的相位重建后台计算引擎
# ==========================================
class PhaseWorker(QThread):
    result_ready = pyqtSignal(object, object, int, float, float, float, float)

    def __init__(self):
        super().__init__()
        self.running = False
        self.is_busy = False
        self.has_new_data = False
        self.data = None

        self.tracking_size = None
        self.prev_peak = None
        self.prev_phase = None
        self.wy = None
        self.wx = None
        self.win_2d = None
        self.y_grid = None
        self.x_grid = None
        self.x_grid_1d = None
        self.y_grid_1d = None
        self.cc = None
        self.cr = None

    def update_data(self, raw_data, cx, cy, dx, dy):
        if not self.is_busy:
            self.data = (raw_data, cx, cy, dx, dy)
            self.has_new_data = True

    def run(self):
        self.running = True
        while self.running:
            if not self.has_new_data or self.data is None:
                time.sleep(0.001)
                continue

            self.is_busy = True
            self.has_new_data = False
            raw_data, cx, cy, dx, dy = self.data
            self.data = None

            try:
                target_size = int(max(dx, dy) * 1.5)
                new_ts = 128
                if target_size > 128: new_ts = 256
                if target_size > 256: new_ts = 512
                if target_size > 512: new_ts = 1024

                if self.tracking_size != new_ts:
                    self.tracking_size = new_ts
                    self.prev_peak = None
                    self.prev_phase = None
                    self.wy = np.hanning(new_ts).astype(np.float32)
                    self.wx = np.hanning(new_ts).astype(np.float32)
                    self.win_2d = self.wy[:, None] * self.wx[None, :]
                    self.y_grid, self.x_grid = np.ogrid[:new_ts, :new_ts]
                    self.x_grid_1d = np.arange(new_ts).reshape(1, new_ts)
                    self.y_grid_1d = np.arange(new_ts).reshape(new_ts, 1)
                    self.cc, self.cr = new_ts // 2, new_ts // 2

                ts = self.tracking_size
                H, W = raw_data.shape

                cx_int, cy_int = int(round(cx)), int(round(cy))
                x1 = max(0, min(W - ts, cx_int - ts // 2))
                y1 = max(0, min(H - ts, cy_int - ts // 2))

                I_raw = raw_data[y1:y1 + ts, x1:x1 + ts].astype(np.float32)
                if I_raw.shape[0] != ts or I_raw.shape[1] != ts:
                    self.is_busy = False
                    continue

                cx_local = cx - x1
                cy_local = cy - y1

                I_centered = I_raw - np.mean(I_raw)
                I_win = I_centered * self.win_2d

                FT = np.fft.fftshift(np.fft.fft2(I_win))
                mag = np.abs(FT)

                if self.prev_peak is None:
                    dc_radius = 8
                    dist_sq_dc = (self.x_grid - self.cc) ** 2 + (self.y_grid - self.cr) ** 2
                    mag[dist_sq_dc < dc_radius ** 2] = 0
                    mag[:, :self.cc + 2] = 0

                    pr_int, pc_int = np.unravel_index(np.argmax(mag), mag.shape)

                    w_s = 3
                    r_sl = slice(max(0, pr_int - w_s), min(ts, pr_int + w_s + 1))
                    c_sl = slice(max(0, pc_int - w_s), min(ts, pc_int + w_s + 1))
                    sc = mag[r_sl, c_sl]
                    yy_c, xx_c = np.mgrid[r_sl, c_sl]
                    sum_sc = np.sum(sc)
                    if sum_sc > 0:
                        pr_sub = np.sum(yy_c * sc) / sum_sc
                        pc_sub = np.sum(xx_c * sc) / sum_sc
                    else:
                        pr_sub, pc_sub = float(pr_int), float(pc_int)
                    self.prev_peak = (pr_sub, pc_sub)
                else:
                    pr_sub, pc_sub = self.prev_peak

                dist_c = np.sqrt((pr_sub - self.cr) ** 2 + (pc_sub - self.cc) ** 2)
                sigma_f = max(4.0, dist_c / 2.0)
                dist_sq_peak = (self.x_grid - pc_sub) ** 2 + (self.y_grid - pr_sub) ** 2
                mask_f = np.exp(-dist_sq_peak / (2 * sigma_f ** 2)).astype(np.float32)

                FT_filtered = FT * mask_f

                shift_y = self.cr - int(round(pr_sub))
                shift_x = self.cc - int(round(pc_sub))
                FT_shifted = np.roll(FT_filtered, (shift_y, shift_x), axis=(0, 1))

                complex_field = np.fft.ifft2(np.fft.ifftshift(FT_shifted))

                dfy = (pr_sub - int(round(pr_sub))) / ts
                dfx = (pc_sub - int(round(pc_sub))) / ts
                carrier = np.exp(-1j * 2 * np.pi * (dfx * self.x_grid_1d + dfy * self.y_grid_1d))

                phase = np.angle(complex_field * carrier).astype(np.float32)

                disp_mask = ((self.x_grid - cx_local) ** 2 / (dx / 2) ** 2 + (self.y_grid - cy_local) ** 2 / (
                            dy / 2) ** 2) <= 1.0

                if self.prev_phase is not None:
                    if np.any(disp_mask):
                        diff_complex = np.mean(np.exp(1j * phase[disp_mask]) * np.exp(-1j * self.prev_phase[disp_mask]))
                        phase -= np.angle(diff_complex)
                        phase = (phase + np.pi) % (2 * np.pi) - np.pi

                self.prev_phase = phase.copy()

                Display_Phase = phase.copy()
                Display_Phase[~disp_mask] = -4.0

                Display_Fringes = I_raw.copy()
                Display_Fringes[~disp_mask] = 0

                self.result_ready.emit(Display_Phase, Display_Fringes, ts, float(cx_local), float(cy_local), float(dx),
                                       float(dy))

            except Exception as e:
                pass
            finally:
                self.is_busy = False

    def stop(self):
        self.running = False
        self.wait()


# ==========================================
# 实时相位重建分析子窗口
# ==========================================
class PhaseReconstructionWindow(QDialog):
    def __init__(self, main_app, parent=None):
        super().__init__(parent)
        self.main_app = main_app
        self.setWindowTitle("实时干涉相位重建系统 (60FPS极致纯净版)")
        self.setStyleSheet(main_app.styleSheet())

        self.main_layout = QHBoxLayout(self)
        self.main_layout.setContentsMargins(11, 11, 11, 11)
        self.main_layout.setSpacing(11)

        def create_cb():
            lbl = QLabel()
            lbl.setFixedWidth(16)
            lbl.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
            lbl.setStyleSheet("border: 1px solid #555;")
            return lbl

        def make_wrapper(widget, cb):
            w = QWidget()
            l = QHBoxLayout(w)
            l.setContentsMargins(0, 0, 0, 0)
            l.setSpacing(4)
            l.addWidget(widget)
            l.addWidget(cb)
            return w

        self.cb_f = create_cb()
        self.cb_p = create_cb()

        self.plot_fringes = pg.PlotWidget(title="原始干涉条纹 (ROI)")
        self.plot_fringes.getViewBox().setAspectLocked(True)
        self.plot_fringes.hideAxis('left')
        self.plot_fringes.hideAxis('bottom')
        self.plot_fringes.setFixedSize(375, 375)
        self.img_fringes = pg.ImageItem(axisOrder='row-major')
        self.plot_fringes.addItem(self.img_fringes)
        self.wrap_f = make_wrapper(self.plot_fringes, self.cb_f)
        self.main_layout.addWidget(self.wrap_f)

        self.plot_phase = pg.PlotWidget(title="重建相位 (绝对平滑去倾斜)")
        self.plot_phase.getViewBox().setAspectLocked(True)
        self.plot_phase.hideAxis('left')
        self.plot_phase.hideAxis('bottom')
        self.plot_phase.setFixedSize(375, 375)
        self.img_phase = pg.ImageItem(axisOrder='row-major')
        self.plot_phase.addItem(self.img_phase)
        self.wrap_p = make_wrapper(self.plot_phase, self.cb_p)
        self.main_layout.addWidget(self.wrap_p)

        self.lbl_fps = QLabel("FPS: 0.0", self)
        self.lbl_fps.setStyleSheet(
            "color: #00FF00; font-weight: bold; font-size: 16px; background: rgba(0,0,0,150); padding: 4px; border-radius: 4px;")
        self.lbl_fps.move(15, 15)
        self.lbl_fps.setAttribute(Qt.WA_TransparentForMouseEvents)

        self.last_fps_time = time.time()
        self.frame_count = 0

        self.update_layout()
        QTimer.singleShot(100, self.main_app.change_colormap)

        # 启动独立的相位解算子线程
        self.worker = PhaseWorker()
        self.worker.result_ready.connect(self.update_plots)
        self.worker.start()

    def sync_colormap(self, cmap, qimg):
        self.img_phase.setColorMap(cmap)
        self.img_fringes.setColorMap(cmap)
        pix = QPixmap.fromImage(qimg)
        self.cb_f.setPixmap(pix)
        self.cb_f.setScaledContents(True)
        self.cb_p.setPixmap(pix)
        self.cb_p.setScaledContents(True)

    def update_layout(self):
        show_fringes = self.main_app.chk_phase_fringes.isChecked()
        if show_fringes:
            self.wrap_f.show()
            self.resize(810, 410)
        else:
            self.wrap_f.hide()
            self.resize(410, 410)
        self.lbl_fps.raise_()

    def process_frame(self, raw_data, cx, cy, dx, dy, is_fitting):
        if not is_fitting or dx == 0 or dy == 0: return
        self.worker.update_data(raw_data, cx, cy, dx, dy)

    def update_plots(self, Display_Phase, Display_Fringes, ts, cx_local, cy_local, dx, dy):
        self.img_phase.setImage(Display_Phase, autoLevels=False, levels=(-np.pi, np.pi), autoDownsample=True)
        if self.main_app.chk_phase_fringes.isChecked():
            self.img_fringes.setImage(Display_Fringes, autoLevels=False, levels=(0, 255), autoDownsample=True)

        view_rx = dx / 2 * 1.1
        view_ry = dy / 2 * 1.1
        self.plot_phase.setXRange(cx_local - view_rx, cx_local + view_rx, padding=0)
        self.plot_phase.setYRange(cy_local - view_ry, cy_local + view_ry, padding=0)
        self.plot_fringes.setXRange(cx_local - view_rx, cx_local + view_rx, padding=0)
        self.plot_fringes.setYRange(cy_local - view_ry, cy_local + view_ry, padding=0)

        self.frame_count += 1
        now = time.time()
        if now - self.last_fps_time > 0.5:
            fps = self.frame_count / (now - self.last_fps_time)
            self.lbl_fps.setText(f"FPS: {fps:.0f}")
            self.frame_count = 0
            self.last_fps_time = now

    def closeEvent(self, event):
        self.worker.stop()
        super().closeEvent(event)


# ==========================================
# 自定义无边框标题栏类
# ==========================================
class CustomTitleBar(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.setFixedHeight(40)
        self.start_pos = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(11, 0, 0, 0)
        layout.setSpacing(0)

        self.title_label = QLabel()
        self.title_label.setStyleSheet("font-weight: bold; font-size: 16px;")
        layout.addWidget(self.title_label)
        layout.addStretch()

        self.btn_minimize = QPushButton("—")
        self.btn_minimize.setFixedSize(50, 40)
        self.btn_minimize.clicked.connect(self.parent.showMinimized)
        layout.addWidget(self.btn_minimize)

        self.btn_maximize = QPushButton("☐")
        self.btn_maximize.setFixedSize(50, 40)
        self.btn_maximize.clicked.connect(self.toggle_maximize)
        layout.addWidget(self.btn_maximize)

        self.btn_close = QPushButton("✕")
        self.btn_close.setFixedSize(50, 40)
        self.btn_close.clicked.connect(self.parent.close)
        layout.addWidget(self.btn_close)

    def apply_theme(self, theme):
        if theme == 'dark':
            self.title_label.setStyleSheet("color: #00A2FF; font-weight: bold; font-size: 16px;")
            btn_style = "QPushButton { background: transparent; border: none; color: #aaa; font-size: 18px; font-weight: bold; } QPushButton:hover { background: #444; color: white; }"
            close_style = "QPushButton { background: transparent; border: none; color: #aaa; font-size: 18px; font-weight: bold; } QPushButton:hover { background: #E81123; color: white; }"
        else:
            self.title_label.setStyleSheet("color: #005A9E; font-weight: bold; font-size: 16px;")
            btn_style = "QPushButton { background: transparent; border: none; color: #555; font-size: 18px; font-weight: bold; } QPushButton:hover { background: #ddd; color: black; }"
            close_style = "QPushButton { background: transparent; border: none; color: #555; font-size: 18px; font-weight: bold; } QPushButton:hover { background: #E81123; color: white; }"

        self.btn_minimize.setStyleSheet(btn_style)
        self.btn_maximize.setStyleSheet(btn_style)
        self.btn_close.setStyleSheet(close_style)

    def toggle_maximize(self):
        if self.parent.isMaximized():
            self.parent.showNormal()
            self.btn_maximize.setText("☐")
        else:
            self.parent.showMaximized()
            self.btn_maximize.setText("❐")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.start_pos = event.globalPos() - self.parent.pos()
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton and self.start_pos:
            self.parent.move(event.globalPos() - self.start_pos)
            event.accept()

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.toggle_maximize()
            event.accept()


# ==========================================
# 重构保存选项弹窗 (新增全选控制与解耦图层控制)
# ==========================================
class SaveDialog(QDialog):
    def __init__(self, main_app):
        super().__init__(main_app)
        self.main_app = main_app
        self.setWindowTitle(main_app.tr("Save Options", "保存选项"))
        self.resize(525, 400)
        self.setStyleSheet(main_app.styleSheet())

        layout = QVBoxLayout(self)

        path_layout = QHBoxLayout()
        self.line_path = QLineEdit()
        if self.main_app.session_save_path:
            self.line_path.setText(self.main_app.session_save_path)
        else:
            base_dir = os.path.abspath(os.path.dirname(__file__))
            date_str = datetime.datetime.now().strftime("%Y-%m-%d")
            default_path = os.path.join(base_dir, "scan", date_str)
            self.line_path.setText(os.path.normpath(default_path))

        self.btn_browse = QPushButton("📂")
        self.btn_browse.setFixedSize(60, 35)
        self.btn_browse.clicked.connect(self.browse_folder)

        path_layout.addWidget(QLabel(main_app.tr("Path:", "保存路径:")))
        path_layout.addWidget(self.line_path)
        path_layout.addWidget(self.btn_browse)
        layout.addLayout(path_layout)

        name_layout = QHBoxLayout()
        self.line_name = QLineEdit()
        self.line_name.setText(self.main_app.session_save_filename)
        name_layout.addWidget(QLabel(main_app.tr("File Name:", "文件名:")))
        name_layout.addWidget(self.line_name)
        layout.addLayout(name_layout)

        self.grp_format = QGroupBox(main_app.tr("Formats", "保存格式"))
        fmt_layout = QHBoxLayout(self.grp_format)
        self.chk_bmp = QCheckBox(".bmp")
        self.chk_png = QCheckBox(".png")
        self.chk_jpg = QCheckBox(".jpg")
        self.chk_csv = QCheckBox("2D (.csv)")
        self.chk_bmp.setChecked(True)
        self.chk_csv.setChecked(True)
        for cb in [self.chk_bmp, self.chk_png, self.chk_jpg, self.chk_csv]:
            fmt_layout.addWidget(cb)
        layout.addWidget(self.grp_format)

        self.grp_overlay = QGroupBox(main_app.tr("Overlay Elements", "保存图层定制"))
        overlay_layout = QVBoxLayout(self.grp_overlay)

        btn_lyout = QHBoxLayout()
        self.btn_sel_all = QPushButton("全选")
        self.btn_sel_none = QPushButton("全不选")
        self.btn_sel_all.clicked.connect(lambda: self.set_overlays(True))
        self.btn_sel_none.clicked.connect(lambda: self.set_overlays(False))
        btn_lyout.addWidget(self.btn_sel_all)
        btn_lyout.addWidget(self.btn_sel_none)
        btn_lyout.addStretch()
        overlay_layout.addLayout(btn_lyout)

        chk_lyout = QHBoxLayout()
        self.chk_save_cross = QCheckBox("十字光标")
        self.chk_save_ring = QCheckBox("拟合光阑环")
        self.chk_save_raw_prof = QCheckBox("原始侧边强度线")
        self.chk_save_fit_prof = QCheckBox("拟合侧边曲线")

        self.chk_save_cross.setChecked(self.main_app.chk_crosshair.isChecked())
        self.chk_save_raw_prof.setChecked(self.main_app.chk_profiles.isChecked())
        is_fitting = self.main_app.chk_d4s.isChecked() or self.main_app.chk_1e2.isChecked() or self.main_app.chk_knife.isChecked()
        self.chk_save_ring.setChecked(is_fitting)
        self.chk_save_fit_prof.setChecked(is_fitting)

        chk_lyout.addWidget(self.chk_save_cross)
        chk_lyout.addWidget(self.chk_save_ring)
        chk_lyout.addWidget(self.chk_save_raw_prof)
        chk_lyout.addWidget(self.chk_save_fit_prof)
        overlay_layout.addLayout(chk_lyout)
        layout.addWidget(self.grp_overlay)

        btn_layout = QHBoxLayout()
        self.btn_confirm = QPushButton("保存")
        self.btn_confirm.clicked.connect(self.accept)
        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.clicked.connect(self.reject)

        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_confirm)
        btn_layout.addWidget(self.btn_cancel)
        layout.addLayout(btn_layout)

    def set_overlays(self, state):
        self.chk_save_cross.setChecked(state)
        self.chk_save_ring.setChecked(state)
        self.chk_save_raw_prof.setChecked(state)
        self.chk_save_fit_prof.setChecked(state)

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择目录", self.line_path.text())
        if folder: self.line_path.setText(os.path.normpath(folder))

    def accept(self):
        save_dir = self.line_path.text()
        file_name = self.line_name.text()
        if not os.path.exists(save_dir): os.makedirs(save_dir, exist_ok=True)
        if not hasattr(self.main_app, 'latest_frame') or self.main_app.latest_frame is None: return

        frame_data = self.main_app.latest_frame
        base_file_path = os.path.join(save_dir, file_name)
        saved_files = []
        self.main_app.session_save_path = save_dir
        self.main_app.session_save_filename = file_name

        try:
            if self.chk_csv.isChecked():
                np.savetxt(base_file_path + ".csv", frame_data, delimiter=",", fmt='%d')
                saved_files.append(base_file_path + ".csv")

            if self.chk_bmp.isChecked() or self.chk_png.isChecked() or self.chk_jpg.isChecked():
                h, w = frame_data.shape
                map_key = self.main_app.combo_colormap.currentData()
                cmap = self.main_app.colormaps_data.get(map_key)

                if cmap is not None:
                    lut = cmap.getLookupTable(0.0, 1.0, 256).astype(np.uint8)
                    colored_frame = lut[frame_data]
                    img_format = QImage.Format_RGBA8888 if colored_frame.shape[2] == 4 else QImage.Format_RGB888
                    qimg = QImage(colored_frame.data, w, h, w * colored_frame.shape[2], img_format).copy()
                else:
                    frame_contig = np.ascontiguousarray(frame_data)
                    qimg = QImage(frame_contig.data, w, h, w, QImage.Format_Grayscale8).copy()

                painter = QPainter(qimg)
                painter.setRenderHint(QPainter.Antialiasing)
                app = self.main_app

                if self.chk_save_cross.isChecked() and hasattr(app, 'h_line') and app.h_line.xData is not None:
                    painter.setPen(QPen(Qt.green, 2, Qt.DashLine))
                    x_c, y_c = app.h_line.xData, app.h_line.yData
                    v_x, v_y = app.v_line.xData, app.v_line.yData
                    if len(x_c) == 4:
                        painter.drawLine(int(x_c[0]), int(y_c[0]), int(x_c[1]), int(y_c[1]))
                        painter.drawLine(int(x_c[2]), int(y_c[2]), int(x_c[3]), int(y_c[3]))
                        painter.drawLine(int(v_x[0]), int(v_y[0]), int(v_x[1]), int(v_y[1]))
                        painter.drawLine(int(v_x[2]), int(v_y[2]), int(v_x[3]), int(v_y[3]))
                    elif len(x_c) == 2:
                        painter.drawLine(int(x_c[0]), int(y_c[0]), int(x_c[1]), int(y_c[1]))
                        painter.drawLine(int(v_x[0]), int(v_y[0]), int(v_x[1]), int(v_y[1]))

                if self.chk_save_ring.isChecked() and hasattr(app, 'beam_ring') and app.beam_ring.xData is not None:
                    painter.setPen(QPen(Qt.white, 4, Qt.SolidLine))
                    r_x, r_y = app.beam_ring.xData, app.beam_ring.yData
                    for i in range(len(r_x) - 1):
                        painter.drawLine(int(r_x[i]), int(r_y[i]), int(r_x[i + 1]), int(r_y[i + 1]))
                    painter.setPen(QPen(Qt.black, 2, Qt.DashLine))
                    for i in range(len(r_x) - 1):
                        painter.drawLine(int(r_x[i]), int(r_y[i]), int(r_x[i + 1]), int(r_y[i + 1]))

                if self.chk_save_raw_prof.isChecked() and hasattr(app, 'curve_x') and app.curve_x.xData is not None:
                    painter.setPen(QPen(Qt.white if app.current_theme == 'dark' else Qt.black, 1, Qt.SolidLine))
                    x1, y1 = app.curve_x.xData, app.curve_x.yData
                    for i in range(len(x1) - 1):
                        painter.drawLine(int(x1[i]), int(y1[i]), int(x1[i + 1]), int(y1[i + 1]))
                    x2, y2 = app.curve_y.xData, app.curve_y.yData
                    for i in range(len(x2) - 1):
                        painter.drawLine(int(x2[i]), int(y2[i]), int(x2[i + 1]), int(y2[i + 1]))

                if self.chk_save_fit_prof.isChecked() and hasattr(app,
                                                                  'fit_curve_x') and app.fit_curve_x.xData is not None:
                    painter.setPen(QPen(Qt.white, 2, Qt.DashLine))
                    x1, y1 = app.fit_curve_x.xData, app.fit_curve_x.yData
                    for i in range(len(x1) - 1):
                        painter.drawLine(int(x1[i]), int(y1[i]), int(x1[i + 1]), int(y1[i + 1]))
                    x2, y2 = app.fit_curve_y.xData, app.fit_curve_y.yData
                    for i in range(len(x2) - 1):
                        painter.drawLine(int(x2[i]), int(y2[i]), int(x2[i + 1]), int(y2[i + 1]))

                painter.end()

                bar_width = max(15, int(w * 0.025))
                margin_width = 8

                if cmap is not None:
                    lut = cmap.getLookupTable(0.0, 1.0, 256).astype(np.uint8)
                    indices = np.linspace(255, 0, h).astype(int)
                    bar_colors = lut[indices]
                    bar_patch = np.tile(bar_colors[:, np.newaxis, :], (1, bar_width, 1))
                    margin_patch = np.zeros((h, margin_width, bar_patch.shape[2]), dtype=np.uint8)
                    if bar_patch.shape[2] == 4: margin_patch[:, :, 3] = 255

                    ptr = qimg.bits()
                    ptr.setsize(qimg.byteCount())
                    drawn_frame = np.array(ptr).reshape(h, w, 4 if qimg.format() == QImage.Format_RGBA8888 else 3)

                    final_frame = np.concatenate((drawn_frame, margin_patch, bar_patch), axis=1)
                    frame_contig = np.ascontiguousarray(final_frame)
                    new_h, new_w, ch = frame_contig.shape
                    qimg_final = QImage(frame_contig.data, new_w, new_h, new_w * ch, qimg.format())
                else:
                    qimg_final = qimg

                if self.chk_bmp.isChecked():
                    qimg_final.save(base_file_path + ".bmp")
                    saved_files.append(base_file_path + ".bmp")
                if self.chk_png.isChecked():
                    qimg_final.save(base_file_path + ".png")
                    saved_files.append(base_file_path + ".png")
                if self.chk_jpg.isChecked():
                    qimg_final.save(base_file_path + ".jpg")
                    saved_files.append(base_file_path + ".jpg")

            QMessageBox.information(self, "成功", f"文件已保存:\n" + "\n".join(saved_files))
            super().accept()
        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"发生错误: \n{str(e)}")


class CaptureDialog(QDialog):
    def __init__(self, main_app):
        super().__init__(main_app)
        self.main_app = main_app
        self.setWindowTitle(main_app.tr("Capture Preview", "图像采集预览"))
        self.resize(900, 675)
        self.setStyleSheet(main_app.styleSheet())
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.plot_widget = pg.PlotWidget(background='#000000')
        self.plot_widget.hideAxis('left')
        self.plot_widget.hideAxis('bottom')
        self.plot_widget.hideAxis('right')
        self.plot_widget.hideAxis('top')
        view_box = self.plot_widget.getViewBox()
        view_box.invertY(True)
        view_box.setAspectLocked(True)
        view_box.setDefaultPadding(0.0)
        self.image_item = pg.ImageItem(axisOrder='row-major')
        self.plot_widget.addItem(self.image_item)
        layout.addWidget(self.plot_widget)
        btn_widget = QWidget()
        btn_layout = QHBoxLayout(btn_widget)
        btn_layout.setContentsMargins(11, 11, 11, 11)
        self.btn_refresh = QPushButton(main_app.tr("Refresh Capture", "刷新采集"))
        self.btn_refresh.clicked.connect(self.refresh_image)
        self.btn_save = QPushButton(main_app.tr("Quick Save", "直接保存"))
        self.btn_save.clicked.connect(self.quick_save)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_refresh)
        btn_layout.addWidget(self.btn_save)
        layout.addWidget(btn_widget)
        self.refresh_image()

    def refresh_image(self):
        if self.main_app.latest_frame is not None:
            self.image_item.setImage(self.main_app.latest_frame, autoLevels=False, levels=(0, 255))
            map_key = self.main_app.combo_colormap.currentData()
            if map_key and map_key in self.main_app.colormaps_data:
                self.image_item.setColorMap(self.main_app.colormaps_data[map_key])

    def quick_save(self):
        dlg = SaveDialog(self.main_app)
        dlg.exec_()


class M2PlotWindow(QDialog):
    def __init__(self, title, z_arr, d_x, d_y, lambda_mm, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"光束质量 (M²) 分析模型 - {title}")
        self.setStyleSheet("background-color: #FFFFFF; color: #000000;")
        self.resize(825, 638)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        old_fg = pg.getConfigOption('foreground')
        old_bg = pg.getConfigOption('background')
        pg.setConfigOption('foreground', 'k')
        pg.setConfigOption('background', 'w')
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setFixedSize(675, 450)
        layout.addWidget(self.plot_widget, alignment=Qt.AlignCenter)
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.btn_save = QPushButton("保存 SCI 格式图像 (Save Image)")
        self.btn_save.setFixedSize(225, 45)
        self.btn_save.setStyleSheet(
            "QPushButton { background-color: #E0E0E0; border: 1px solid black; font-family: 'Times New Roman'; font-weight: bold; font-size: 14px; color: black; } QPushButton:hover { background-color: #CCCCCC; } QPushButton:pressed { background-color: #AAAAAA; }")
        self.btn_save.clicked.connect(self.save_image)
        btn_layout.addWidget(self.btn_save)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        plot = self.plot_widget.getPlotItem()
        coefs_raw_x = np.polyfit(z_arr, np.array(d_x) ** 2, 2)
        z_shift = -coefs_raw_x[1] / (2.0 * coefs_raw_x[0]) if coefs_raw_x[0] != 0 else 0
        z_new = z_arr - z_shift
        d_x_mm = np.array(d_x) / 1000.0
        d_y_mm = np.array(d_y) / 1000.0
        coefs_x = np.polyfit(z_new, d_x_mm ** 2, 2)
        coefs_y = np.polyfit(z_new, d_y_mm ** 2, 2)

        def calc_m2(coefs):
            C, B, A = coefs
            val = A * C - (B ** 2) / 4.0
            if val <= 0 or C <= 0: return float('nan')
            return (np.pi / (4.0 * lambda_mm)) * np.sqrt(val)

        m2_x = calc_m2(coefs_x)
        m2_y = calc_m2(coefs_y)
        font = QFont("Times New Roman", 16, QFont.Bold)
        pen_axis = pg.mkPen('k', width=3)
        for axis_name in ['left', 'bottom', 'top', 'right']:
            plot.showAxis(axis_name)
            axis = plot.getAxis(axis_name)
            axis.setPen(pen_axis)
            if axis_name in ['top', 'right']:
                axis.setStyle(showValues=False, tickLength=0)
            else:
                axis.setTextPen('k')
                axis.setTickFont(font)
                axis.setStyle(tickLength=-10, tickTextOffset=15)
        x_range = max(z_new) - min(z_new)
        if x_range == 0: x_range = 100
        step_x = x_range / 6.0
        p_x = np.power(10.0, np.floor(np.log10(step_x)))
        if step_x / p_x >= 5:
            major_x = 5 * p_x
        elif step_x / p_x >= 2:
            major_x = 2 * p_x
        else:
            major_x = p_x
        minor_x = major_x / 2.0
        plot.getAxis('bottom').setTickSpacing(levels=[(major_x, 0), (minor_x, 0)])
        plot.getAxis('top').setTickSpacing(levels=[(major_x, 0), (minor_x, 0)])
        y_max_val = max(max(d_x), max(d_y))
        y_min_val = min(min(d_x), min(d_y))
        y_range = y_max_val - y_min_val
        if y_range == 0: y_range = 100
        step_y = y_range / 6.0
        p_y = np.power(10.0, np.floor(np.log10(step_y)))
        if step_y / p_y >= 5:
            major_y = 5 * p_y
        elif step_y / p_y >= 2:
            major_y = 2 * p_y
        else:
            major_y = p_y
        minor_y = major_y / 2.0
        plot.getAxis('left').setTickSpacing(levels=[(major_y, 0), (minor_y, 0)])
        plot.getAxis('right').setTickSpacing(levels=[(major_y, 0), (minor_y, 0)])
        plot.getAxis('bottom').setLabel(text='Relative Position Z (mm)', color='k',
                                        **{'font-family': 'Times New Roman', 'font-size': '18pt',
                                           'font-weight': 'bold'})
        plot.getAxis('left').setLabel(text='Beam Diameter (\u03bcm)', color='k',
                                      **{'font-family': 'Times New Roman', 'font-size': '18pt', 'font-weight': 'bold'})
        legend = plot.addLegend(offset=(15, 15))
        legend.setPen(pg.mkPen('k', width=2))
        legend.setBrush(pg.mkBrush('w'))
        legend.setLabelTextSize('14pt')
        plot.plot(z_new, d_x, pen=None, symbol='o', symbolPen='r', symbolBrush='r', symbolSize=9, name='X Direction')
        plot.plot(z_new, d_y, pen=None, symbol='s', symbolPen='b', symbolBrush='b', symbolSize=9, name='Y Direction')
        z_smooth = np.linspace(min(z_new) - 10, max(z_new) + 10, 200)
        if not np.isnan(m2_x):
            d_fit_x = np.sqrt(np.maximum(0, coefs_x[2] + coefs_x[1] * z_smooth + coefs_x[0] * z_smooth ** 2)) * 1000.0
            plot.plot(z_smooth, d_fit_x, pen=pg.mkPen('r', width=2, style=Qt.SolidLine))
        if not np.isnan(m2_y):
            d_fit_y = np.sqrt(np.maximum(0, coefs_y[2] + coefs_y[1] * z_smooth + coefs_y[0] * z_smooth ** 2)) * 1000.0
            plot.plot(z_smooth, d_fit_y, pen=pg.mkPen('b', width=2, style=Qt.SolidLine))
        res_html = f'<div style="text-align: center; font-family: Times New Roman; font-size: 20pt; font-weight: bold; color: black; line-height: 1.2;">M<sub>x</sub>² = {m2_x:.2f}<br>M<sub>y</sub>² = {m2_y:.2f}</div>'
        text_item = pg.TextItem(html=res_html, anchor=(0.5, 0.5))
        plot.addItem(text_item)
        y_pos = y_min_val + 0.75 * y_range
        x_pos = (min(z_new) + max(z_new)) / 2.0
        text_item.setPos(x_pos, y_pos)
        pg.setConfigOption('foreground', old_fg)
        pg.setConfigOption('background', old_bg)

    def save_image(self):
        exporter = pyqtgraph.exporters.ImageExporter(self.plot_widget.getPlotItem())
        exporter.parameters()['width'] = 2000
        path, _ = QFileDialog.getSaveFileName(self, "Save SCI Image", "", "PNG Files (*.png);;JPEG Files (*.jpg)")
        if path:
            exporter.export(path)
            QMessageBox.information(self, "Success", "SCI 级别绘图已成功导出。")


class BeamGageApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)
        # 整体横纵等比例缩小一半，面积约为原来的 1/4
        self.resize(1125, 750)

        self.current_lang = 'zh'
        self.current_theme = 'dark'
        self.pixel_size_um = 5.86

        self.is_running = False
        self.is_paused = False
        self.is_connected = False
        self.device_list = []
        self.connection_failed = False

        self.latest_frame = None
        self.current_frame = None
        self.current_fps = 0.0
        self.current_actual_exp = 10.0

        self.force_auto_range = False

        self.cached_shape = None
        self.x_indices = None
        self.y_indices = None

        self.m2_data_list = []
        self.current_sig_x = 0.0
        self.current_sig_y = 0.0
        self.m2_plot_windows = []

        self.phase_window = None

        self.h_camera = None
        self.camera_thread = None

        self.ui_render_timer = QTimer()
        self.ui_render_timer.timeout.connect(self.render_process_loop)

        self.sim_timer = QTimer()
        self.sim_timer.timeout.connect(self.update_sim_frame)
        self.sim_phase = 0.0
        self.sim_ideal_profile_x = None
        self.sim_ideal_profile_y = None

        self.colormaps_data = self.create_custom_colormaps()
        self.init_ui()
        self.apply_theme()
        self.retranslate_ui()

        self.session_save_path = None
        self.session_save_filename = "BeamData_001"

    def closeEvent(self, event):
        self.stop_real_camera()
        if self.is_connected and self.h_camera is not None:
            try:
                mvdll.CameraUnInit(self.h_camera)
            except:
                pass
        if hasattr(self, 'phase_window') and self.phase_window is not None:
            self.phase_window.close()
        event.accept()

    def tr(self, en_text, zh_text):
        return zh_text if self.current_lang == 'zh' else en_text

    def create_custom_colormaps(self):
        maps = {}
        pos_jet = np.array([0.0, 0.115, 0.225, 0.345, 0.655, 0.775, 0.885, 1.0])
        color_jet = np.array([
            [0, 0, 143, 255], [0, 0, 255, 255], [0, 119, 255, 255], [0, 255, 255, 255],
            [255, 255, 0, 255], [255, 127, 0, 255], [255, 0, 0, 255], [127, 0, 0, 255]
        ], dtype=np.ubyte)
        maps["jet"] = pg.ColorMap(pos_jet, color_jet)

        pos_parula = np.array([0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0])
        color_parula = np.array([
            [53, 42, 135, 255], [4, 84, 211, 255], [5, 140, 204, 255],
            [32, 163, 134, 255], [115, 182, 107, 255], [210, 184, 53, 255],
            [253, 205, 36, 255], [240, 249, 33, 255]
        ], dtype=np.ubyte)
        maps["parula"] = pg.ColorMap(pos_parula, color_parula)

        pos_viridis = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        color_viridis = np.array([
            [68, 1, 84, 255], [59, 82, 139, 255], [33, 145, 140, 255],
            [94, 201, 98, 255], [253, 231, 37, 255]
        ], dtype=np.ubyte)
        maps["viridis"] = pg.ColorMap(pos_viridis, color_viridis)

        pos_hsv = np.array([0.0, 0.16, 0.33, 0.5, 0.66, 0.83, 1.0])
        color_hsv = np.array([
            [255, 0, 0, 255], [255, 255, 0, 255], [0, 255, 0, 255],
            [0, 255, 255, 255], [0, 0, 255, 255], [255, 0, 255, 255],
            [255, 0, 0, 255]
        ], dtype=np.ubyte)
        maps["hsv"] = pg.ColorMap(pos_hsv, color_hsv)

        pos_hot = np.array([0.0, 0.375, 0.75, 1.0])
        color_hot = np.array([
            [0, 0, 0, 255], [255, 0, 0, 255],
            [255, 255, 0, 255], [255, 255, 255, 255]
        ], dtype=np.ubyte)
        maps["hot"] = pg.ColorMap(pos_hot, color_hot)

        pos_bone = np.array([0.0, 0.375, 0.75, 1.0])
        color_bone = np.array([
            [0, 0, 0, 255], [84, 84, 116, 255],
            [168, 199, 199, 255], [255, 255, 255, 255]
        ], dtype=np.ubyte)
        maps["bone"] = pg.ColorMap(pos_bone, color_bone)

        pos_db = np.array([0.0, 0.5, 1.0])
        color_db = np.array([
            [1, 1, 1, 255],
            [10, 58, 115, 255],
            [17, 113, 190, 255]
        ], dtype=np.ubyte)
        maps["darkblue"] = pg.ColorMap(pos_db, color_db)

        pos_rb = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        color_rb = np.array([
            [51, 0, 51, 255],
            [220, 0, 150, 255],
            [0, 0, 255, 255],
            [0, 255, 128, 255],
            [255, 255, 0, 255],
            [255, 0, 0, 255]
        ], dtype=np.ubyte)
        maps["rainbow"] = pg.ColorMap(pos_rb, color_rb)

        pos_vi = np.array([0.0, 1.0])
        color_vi = np.array([
            [1, 1, 1, 255],
            [192, 92, 251, 255]
        ], dtype=np.ubyte)
        maps["violet"] = pg.ColorMap(pos_vi, color_vi)

        maps["green"] = pg.ColorMap(np.array([0.0, 1.0]),
                                    np.array([[1, 1, 1, 255], [0, 255, 0, 255]], dtype=np.ubyte))
        maps["red"] = pg.ColorMap(np.array([0.0, 1.0]),
                                  np.array([[1, 1, 1, 255], [255, 0, 0, 255]], dtype=np.ubyte))

        return maps

    def init_ui(self):
        font_big = QFont()
        font_big.setPointSize(11)  # 字号缩小
        font_big.setBold(True)

        self.top_widget = QWidget()
        top_layout = QVBoxLayout(self.top_widget)
        top_layout.setContentsMargins(1, 1, 1, 1)
        top_layout.setSpacing(0)
        self.setCentralWidget(self.top_widget)

        self.title_bar = CustomTitleBar(self)
        top_layout.addWidget(self.title_bar)

        self.content_widget = QWidget()
        main_layout = QVBoxLayout(self.content_widget)
        main_layout.setContentsMargins(8, 8, 8, 8)
        top_layout.addWidget(self.content_widget)

        self.ribbon = QTabWidget()
        self.ribbon.setMinimumHeight(260)  # 高度缩小
        self.ribbon.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # --- Tab 1: Capture ---
        self.tab_capture = QWidget()
        layout_capture = QHBoxLayout(self.tab_capture)
        layout_capture.setAlignment(Qt.AlignLeft)
        layout_capture.setContentsMargins(11, 11, 11, 16)

        self.grp_device_mgr = QGroupBox()
        l_device_mgr = QVBoxLayout(self.grp_device_mgr)

        self.btn_scan = QPushButton()
        self.btn_scan.clicked.connect(self.scan_devices)
        self.btn_connect = QPushButton()
        self.btn_connect.clicked.connect(self.connect_device)
        self.combo_devices = QComboBox()
        self.combo_devices.activated.connect(self.connect_device)

        l_device_mgr.addWidget(self.btn_scan)
        l_device_mgr.addWidget(self.btn_connect)
        l_device_mgr.addWidget(self.combo_devices)

        self.grp_control = QGroupBox()
        l_control = QVBoxLayout(self.grp_control)
        self.btn_pause = QPushButton()
        self.btn_pause.setCheckable(True)
        self.btn_pause.setFixedSize(90, 90)  # 尺寸缩小
        self.btn_pause.clicked.connect(self.toggle_pause)
        l_control.addWidget(self.btn_pause, alignment=Qt.AlignCenter)

        self.grp_exposure = QGroupBox()
        l_exposure_main = QHBoxLayout(self.grp_exposure)

        l_auto = QVBoxLayout()
        self.chk_auto_exp = QCheckBox()
        self.chk_auto_exp.setChecked(True)
        l_auto.addStretch()
        l_auto.addWidget(self.chk_auto_exp)
        l_auto.addStretch()
        l_exposure_main.addLayout(l_auto)

        l_sliders = QVBoxLayout()
        l_sliders.setContentsMargins(16, 0, 0, 0)
        l_sliders.setSpacing(16)

        h_exp = QHBoxLayout()
        self.lbl_exp = QLabel()
        self.lbl_exp.setFixedWidth(50)
        self.slider_exp = QSlider(Qt.Horizontal)
        self.slider_exp.setRange(0, 101885951)
        self.slider_exp.setEnabled(False)
        self.spin_exp = QDoubleSpinBox()
        self.spin_exp.setRange(0.0049, 10188.6000)
        self.spin_exp.setDecimals(4)
        self.spin_exp.setSingleStep(0.0001)
        self.spin_exp.setValue(1.0)
        self.spin_exp.setEnabled(False)
        self.spin_exp.setFixedWidth(90)
        lbl_unit_exp = QLabel("ms")
        lbl_unit_exp.setFixedWidth(25)
        h_exp.addWidget(self.lbl_exp)
        h_exp.addWidget(self.slider_exp)
        h_exp.addWidget(self.spin_exp)
        h_exp.addWidget(lbl_unit_exp)

        h_gain = QHBoxLayout()
        self.lbl_gain = QLabel()
        self.lbl_gain.setFixedWidth(50)
        self.slider_gain = QSlider(Qt.Horizontal)
        self.slider_gain.setRange(0, 100)
        self.slider_gain.setValue(0)
        self.spin_gain = QSpinBox()
        self.spin_gain.setRange(0, 100)
        self.spin_gain.setSingleStep(1)
        self.spin_gain.setValue(0)
        self.spin_gain.setFixedWidth(90)
        lbl_unit_gain = QLabel("%")
        lbl_unit_gain.setFixedWidth(25)
        h_gain.addWidget(self.lbl_gain)
        h_gain.addWidget(self.slider_gain)
        h_gain.addWidget(self.spin_gain)
        h_gain.addWidget(lbl_unit_gain)

        l_sliders.addLayout(h_exp)
        l_sliders.addLayout(h_gain)
        l_exposure_main.addLayout(l_sliders)

        self.slider_exp.valueChanged.connect(self.on_slider_exp_changed)
        self.spin_exp.valueChanged.connect(self.on_spin_exp_changed)
        self.slider_gain.valueChanged.connect(self.spin_gain.setValue)
        self.spin_gain.valueChanged.connect(self.slider_gain.setValue)
        self.slider_gain.valueChanged.connect(self.on_gain_changed)
        self.chk_auto_exp.stateChanged.connect(self.on_auto_exp_changed)

        self.grp_actions = QGroupBox()
        l_actions = QVBoxLayout(self.grp_actions)
        h_action_top = QHBoxLayout()
        self.btn_capture = QPushButton()
        self.btn_save = QPushButton()
        self.btn_capture.clicked.connect(self.open_capture_dialog)
        self.btn_save.clicked.connect(self.open_save_dialog)
        h_action_top.addWidget(self.btn_capture)
        h_action_top.addWidget(self.btn_save)
        self.btn_bg_sub = QPushButton()
        self.btn_bg_sub.setCheckable(True)
        l_actions.addLayout(h_action_top)
        l_actions.addWidget(self.btn_bg_sub)

        layout_capture.addWidget(self.grp_device_mgr)
        layout_capture.addWidget(self.grp_control)
        layout_capture.addWidget(self.grp_exposure)
        layout_capture.addWidget(self.grp_actions)
        self.ribbon.addTab(self.tab_capture, "")

        # --- Tab 2: Display ---
        self.tab_display = QWidget()
        layout_display = QHBoxLayout(self.tab_display)
        layout_display.setAlignment(Qt.AlignLeft)
        layout_display.setContentsMargins(11, 11, 11, 16)

        self.grp_color = QGroupBox()
        l_color = QHBoxLayout(self.grp_color)
        self.lbl_cmap = QLabel()
        self.combo_colormap = QComboBox()
        self.combo_colormap.currentIndexChanged.connect(self.change_colormap)
        l_color.addWidget(self.lbl_cmap)
        l_color.addWidget(self.combo_colormap)

        self.grp_2d = QGroupBox()
        l_2d = QHBoxLayout(self.grp_2d)
        self.chk_crosshair = QCheckBox()
        self.chk_crosshair.setChecked(True)
        self.chk_crosshair.stateChanged.connect(self.toggle_crosshair)
        self.chk_profiles = QCheckBox()
        self.chk_profiles.setChecked(False)
        self.chk_profiles.stateChanged.connect(self.toggle_profiles)
        l_2d.addWidget(self.chk_crosshair)
        l_2d.addWidget(self.chk_profiles)

        layout_display.addWidget(self.grp_color)
        layout_display.addWidget(self.grp_2d)
        self.ribbon.addTab(self.tab_display, "")

        # --- Tab 3: Computations (含相位重建) ---
        self.tab_aperture = QWidget()
        layout_aperture = QHBoxLayout(self.tab_aperture)
        layout_aperture.setAlignment(Qt.AlignLeft)
        layout_aperture.setContentsMargins(11, 11, 11, 16)

        self.grp_roi = QGroupBox()
        self.grp_roi.setMinimumWidth(140)
        l_roi = QVBoxLayout(self.grp_roi)

        self.btn_circ = QRadioButton()
        self.btn_sq = QRadioButton()
        self.btn_rect = QRadioButton()
        self.btn_ell = QRadioButton()

        l_roi.addStretch()
        for btn in [self.btn_circ, self.btn_sq, self.btn_rect, self.btn_ell]:
            btn.setAutoExclusive(False)
            l_roi.addWidget(btn)
            l_roi.addStretch()

        self.btn_circ.clicked.connect(lambda: self.toggle_aperture('circle'))
        self.btn_sq.clicked.connect(lambda: self.toggle_aperture('square'))
        self.btn_rect.clicked.connect(lambda: self.toggle_aperture('rect'))
        self.btn_ell.clicked.connect(lambda: self.toggle_aperture('ellipse'))
        layout_aperture.addWidget(self.grp_roi)

        self.grp_calc = QGroupBox()
        l_calc = QVBoxLayout(self.grp_calc)
        l_calc.setSpacing(4)
        l_calc.setContentsMargins(16, 6, 16, 6)

        self.chk_d4s = QCheckBox("D4σ")
        self.chk_1e2 = QCheckBox("1/e²")
        self.chk_knife = QCheckBox()

        self.chk_d4s.clicked.connect(lambda: self.toggle_fit_method(self.chk_d4s))
        self.chk_1e2.clicked.connect(lambda: self.toggle_fit_method(self.chk_1e2))
        self.chk_knife.clicked.connect(lambda: self.toggle_fit_method(self.chk_knife))

        l_calc.addWidget(self.chk_d4s)
        l_calc.addWidget(self.chk_1e2)
        l_calc.addWidget(self.chk_knife)
        layout_aperture.addWidget(self.grp_calc)

        self.grp_sim = QGroupBox()
        l_sim_main = QHBoxLayout(self.grp_sim)

        l_sim_switch = QVBoxLayout()
        self.btn_sim_mode = QPushButton()
        self.btn_sim_mode.setCheckable(True)
        self.btn_sim_mode.setFixedSize(150, 45)
        self.btn_sim_mode.clicked.connect(self.on_sim_mode_toggled)

        self.chk_sim_jitter = QCheckBox()
        self.chk_sim_jitter.setChecked(True)

        self.chk_sim_interf = QCheckBox(self.tr("Interference", "平面波干涉"))

        l_sim_switch.addStretch()
        l_sim_switch.addWidget(self.btn_sim_mode, alignment=Qt.AlignCenter)
        l_sim_switch.addWidget(self.chk_sim_jitter, alignment=Qt.AlignCenter)
        l_sim_switch.addWidget(self.chk_sim_interf, alignment=Qt.AlignCenter)
        l_sim_switch.addStretch()
        l_sim_main.addLayout(l_sim_switch)

        self.sim_options_widget = QWidget()
        l_opts = QVBoxLayout(self.sim_options_widget)
        l_opts.setContentsMargins(11, 0, 11, 0)
        l_opts.setSpacing(4)

        self.radio_gauss = QRadioButton()
        self.radio_sgauss = QRadioButton()
        self.radio_hgauss = QRadioButton()
        self.radio_lgauss = QRadioButton()
        self.radio_gauss.setChecked(True)

        for rb in [self.radio_gauss, self.radio_sgauss, self.radio_hgauss, self.radio_lgauss]:
            rb.clicked.connect(self.on_beam_type_changed)
            l_opts.addWidget(rb)

        l_sim_main.addWidget(self.sim_options_widget)

        self.sim_params_widget = QWidget()
        l_params = QFormLayout(self.sim_params_widget)
        l_params.setContentsMargins(6, 0, 6, 0)
        l_params.setVerticalSpacing(4)

        self.lbl_param1 = QLabel("Param 1:")
        self.spin_param1 = QSpinBox()
        self.spin_param1.setRange(0, 20)

        self.lbl_param2 = QLabel("Param 2:")
        self.spin_param2 = QSpinBox()
        self.spin_param2.setRange(0, 20)

        l_params.addRow(self.lbl_param1, self.spin_param1)
        l_params.addRow(self.lbl_param2, self.spin_param2)

        l_sim_main.addWidget(self.sim_params_widget)
        layout_aperture.addWidget(self.grp_sim)

        self.grp_phase = QGroupBox(self.tr("Phase Retrieval", "相位重建模块"))
        l_phase = QVBoxLayout(self.grp_phase)

        self.btn_phase_start = QPushButton(self.tr("Start Reconstruction", "开始重建"))
        self.btn_phase_start.setCheckable(True)
        self.btn_phase_start.setFont(font_big)
        self.btn_phase_start.clicked.connect(self.toggle_phase_window)

        h_phase_opts = QHBoxLayout()
        self.chk_phase_fringes = QCheckBox(self.tr("Fringes", "干涉条纹"))

        h_phase_opts.addWidget(self.chk_phase_fringes)
        self.chk_phase_fringes.stateChanged.connect(self.sync_phase_layout)

        l_phase.addWidget(self.btn_phase_start)
        l_phase.addLayout(h_phase_opts)
        layout_aperture.addWidget(self.grp_phase)

        self.ribbon.addTab(self.tab_aperture, "")

        # --- Tab 4: Beam Quality (M²) ---
        self.tab_m2 = QWidget()
        layout_m2 = QHBoxLayout(self.tab_m2)
        layout_m2.setAlignment(Qt.AlignLeft)
        layout_m2.setContentsMargins(11, 11, 11, 16)

        self.grp_m2 = QGroupBox()
        h_m2_main = QHBoxLayout(self.grp_m2)
        h_m2_main.setSpacing(22)

        col1 = QVBoxLayout()
        self.btn_m2_mode = QPushButton()
        self.btn_m2_mode.setCheckable(True)
        self.btn_m2_mode.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.btn_m2_mode.setFont(font_big)
        self.btn_m2_mode.clicked.connect(self.toggle_m2_mode)
        col1.addWidget(self.btn_m2_mode)

        col2 = QVBoxLayout()
        col2.addStretch()
        form_m2 = QFormLayout()
        form_m2.setVerticalSpacing(16)

        self.spin_wave = QDoubleSpinBox()
        self.spin_wave.setRange(1, 10000)
        self.spin_wave.setValue(1064.0)
        self.lbl_wave = QLabel()
        form_m2.addRow(self.lbl_wave, self.spin_wave)

        self.spin_focal = QDoubleSpinBox()
        self.spin_focal.setRange(1, 10000)
        self.spin_focal.setValue(100.0)
        h_focal = QHBoxLayout()
        h_focal.addWidget(self.spin_focal)
        self.btn_focal_confirm = QPushButton()
        self.btn_focal_confirm.setMinimumHeight(30)
        h_focal.addWidget(self.btn_focal_confirm)
        self.lbl_focal = QLabel()
        form_m2.addRow(self.lbl_focal, h_focal)

        col2.addLayout(form_m2)
        col2.addStretch()

        col3 = QVBoxLayout()
        col3.setSpacing(11)
        self.btn_m2_record = QPushButton()
        self.btn_m2_record.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.btn_m2_record.clicked.connect(self.record_m2_data)

        self.btn_m2_delete = QPushButton()
        self.btn_m2_delete.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.btn_m2_delete.clicked.connect(self.delete_m2_data)

        col3.addWidget(self.btn_m2_record)
        col3.addWidget(self.btn_m2_delete)

        col4 = QVBoxLayout()
        self.btn_m2_calc = QPushButton()
        self.btn_m2_calc.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.btn_m2_calc.setFont(font_big)
        self.btn_m2_calc.clicked.connect(self.calc_plot_m2)
        col4.addWidget(self.btn_m2_calc)

        h_m2_main.addLayout(col1, stretch=1)
        h_m2_main.addLayout(col2, stretch=1)
        h_m2_main.addLayout(col3, stretch=1)
        h_m2_main.addLayout(col4, stretch=1)

        layout_m2.addWidget(self.grp_m2, stretch=1)
        self.ribbon.addTab(self.tab_m2, "")

        # --- Tab 5: About ---
        self.tab_about = QWidget()
        layout_about = QHBoxLayout(self.tab_about)
        layout_about.setAlignment(Qt.AlignLeft)
        layout_about.setContentsMargins(11, 11, 11, 16)

        self.grp_lang = QGroupBox()
        l_lang = QHBoxLayout(self.grp_lang)
        self.combo_lang = QComboBox()
        self.combo_lang.addItems(["中文 (简体)", "English"])
        self.combo_lang.setCurrentIndex(0 if self.current_lang == 'zh' else 1)
        self.combo_lang.currentIndexChanged.connect(self.switch_language)
        l_lang.addWidget(self.combo_lang)
        layout_about.addWidget(self.grp_lang)

        self.grp_theme = QGroupBox()
        l_theme = QHBoxLayout(self.grp_theme)
        self.combo_theme = QComboBox()
        self.combo_theme.addItems(["深色模式 (Dark)", "浅色模式 (Light)"])
        self.combo_theme.setCurrentIndex(0 if self.current_theme == 'dark' else 1)
        self.combo_theme.currentIndexChanged.connect(self.switch_theme)
        l_theme.addWidget(self.combo_theme)
        layout_about.addWidget(self.grp_theme)

        self.grp_info = QGroupBox()
        l_info = QVBoxLayout(self.grp_info)
        self.lbl_app_name = QLabel()
        self.lbl_app_name.setStyleSheet("font-weight: bold; color: #00A2FF;")
        self.lbl_version = QLabel("Version: 25.0 (Strictly Targeted Interlock Edition)")
        l_info.addWidget(self.lbl_app_name)
        l_info.addWidget(self.lbl_version)
        layout_about.addWidget(self.grp_info)

        self.ribbon.addTab(self.tab_about, "")
        main_layout.addWidget(self.ribbon)

        # ==============================================================
        # 下方工作区
        # ==============================================================
        workspace_layout = QHBoxLayout()
        workspace_layout.setContentsMargins(0, 0, 0, 0)
        workspace_layout.setSpacing(6)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.hideAxis('left')
        self.plot_widget.hideAxis('bottom')
        self.plot_widget.hideAxis('right')
        self.plot_widget.hideAxis('top')
        self.plot_widget.getViewBox().invertY(True)
        self.plot_widget.getViewBox().setAspectLocked(True)

        self.image_item = pg.ImageItem(axisOrder='row-major')
        self.plot_widget.addItem(self.image_item)

        self.colorbar_label = QLabel()
        self.colorbar_label.setFixedWidth(16)
        self.colorbar_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)

        dash_pattern = [6, 5]

        self.pen_cross = pg.mkPen('g', width=2, style=Qt.CustomDashLine)
        self.pen_cross.setDashPattern(dash_pattern)

        self.v_line = pg.PlotCurveItem(pen=self.pen_cross)
        self.h_line = pg.PlotCurveItem(pen=self.pen_cross)
        self.plot_widget.addItem(self.v_line)
        self.plot_widget.addItem(self.h_line)

        self.curve_x = pg.PlotCurveItem()
        self.curve_x.setZValue(10)
        self.curve_y = pg.PlotCurveItem()
        self.curve_y.setZValue(10)
        self.plot_widget.addItem(self.curve_x)
        self.plot_widget.addItem(self.curve_y)
        self.curve_x.setVisible(False)
        self.curve_y.setVisible(False)

        # ★ 拟合线全部改成白色虚线
        self.pen_fit = pg.mkPen('w', width=2, style=Qt.CustomDashLine)
        self.pen_fit.setDashPattern(dash_pattern)
        self.pen_ring = pg.mkPen(color='w', width=2.5, style=Qt.CustomDashLine)
        self.pen_ring.setDashPattern(dash_pattern)

        self.fit_curve_x = pg.PlotCurveItem(pen=self.pen_fit)
        self.fit_curve_y = pg.PlotCurveItem(pen=self.pen_fit)
        self.fit_curve_x.setZValue(11)
        self.fit_curve_y.setZValue(11)
        self.beam_ring = pg.PlotCurveItem(pen=self.pen_ring)
        self.beam_ring.setZValue(15)

        self.plot_widget.addItem(self.fit_curve_x)
        self.plot_widget.addItem(self.fit_curve_y)
        self.plot_widget.addItem(self.beam_ring)

        self.rois = {
            'circle': pg.CircleROI([0, 0], [300, 300], pen=pg.mkPen('r', width=3)),
            'square': pg.ROI([0, 0], [300, 300], pen=pg.mkPen('r', width=3)),
            'rect': pg.RectROI([0, 0], [300, 300], pen=pg.mkPen('r', width=3)),
            'ellipse': pg.EllipseROI([0, 0], [300, 300], pen=pg.mkPen('r', width=3))
        }
        self.rois['square'].addScaleHandle([1, 1], [0, 0])
        self.current_roi_type = None

        for roi in self.rois.values():
            roi.setZValue(20)
            self.plot_widget.addItem(roi)
            roi.hide()
            roi.sigRegionChanged.connect(self.on_roi_changed)

        workspace_layout.addWidget(self.plot_widget, stretch=4)
        workspace_layout.addWidget(self.colorbar_label)

        # --- 右侧数据面板 ---
        self.right_panel = QWidget()
        right_layout = QVBoxLayout(self.right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self.right_panel.setFixedWidth(390)

        self.right_stacked = QStackedWidget()

        self.right_normal_widget = QWidget()
        normal_layout = QVBoxLayout(self.right_normal_widget)
        normal_layout.setContentsMargins(0, 0, 0, 0)

        self.grp_device_params = QGroupBox()
        layout_dev = QFormLayout(self.grp_device_params)
        layout_dev.setContentsMargins(16, 22, 16, 16)

        self.lbl_t_model = QLabel()
        self.lbl_t_conn_status = QLabel()
        self.lbl_t_cap_status = QLabel()
        self.lbl_t_fps = QLabel()
        self.lbl_t_exp_res = QLabel()
        self.lbl_t_gain_res = QLabel()
        self.lbl_t_noise = QLabel()

        self.lbl_model = QLabel("No Device")
        self.lbl_conn_status = QLabel()
        self.lbl_cap_status = QLabel()
        self.lbl_fps = QLabel("0.0")
        self.lbl_exp_res = QLabel("0 us")
        self.lbl_gain_res = QLabel("0 %")
        self.lbl_noise = QLabel("0.0 dB")

        self.dev_val_labels = [self.lbl_model, self.lbl_conn_status, self.lbl_cap_status, self.lbl_fps,
                               self.lbl_exp_res, self.lbl_gain_res, self.lbl_noise]
        self.val_font = QFont("Consolas", 16, QFont.Bold)

        layout_dev.addRow(self.lbl_t_model, self.lbl_model)
        layout_dev.addRow(self.lbl_t_conn_status, self.lbl_conn_status)
        layout_dev.addRow(self.lbl_t_cap_status, self.lbl_cap_status)
        layout_dev.addRow(self.lbl_t_fps, self.lbl_fps)
        layout_dev.addRow(self.lbl_t_exp_res, self.lbl_exp_res)
        layout_dev.addRow(self.lbl_t_gain_res, self.lbl_gain_res)
        layout_dev.addRow(self.lbl_t_noise, self.lbl_noise)

        self.grp_calc_params = QGroupBox()
        layout_calc = QFormLayout(self.grp_calc_params)
        layout_calc.setContentsMargins(16, 22, 16, 16)

        self.lbl_t_peak = QLabel()
        self.lbl_t_centroid = QLabel()
        self.lbl_t_width_x = QLabel()
        self.lbl_t_width_y = QLabel()
        self.lbl_t_aperture = QLabel()

        self.lbl_peak = QLabel("0")
        self.lbl_centroid = QLabel("(0.0, 0.0)")
        self.lbl_width_x = QLabel("0.0 um")
        self.lbl_width_y = QLabel("0.0 um")
        self.lbl_aperture = QLabel("N/A")

        self.calc_val_labels = [self.lbl_peak, self.lbl_centroid, self.lbl_width_x, self.lbl_width_y, self.lbl_aperture]

        layout_calc.addRow(self.lbl_t_peak, self.lbl_peak)
        layout_calc.addRow(self.lbl_t_centroid, self.lbl_centroid)
        layout_calc.addRow(self.lbl_t_width_x, self.lbl_width_x)
        layout_calc.addRow(self.lbl_t_width_y, self.lbl_width_y)
        layout_calc.addRow(self.lbl_t_aperture, self.lbl_aperture)

        normal_layout.addWidget(self.grp_device_params)
        normal_layout.addWidget(self.grp_calc_params)
        normal_layout.addStretch()

        self.right_m2_widget = QWidget()
        m2_layout = QVBoxLayout(self.right_m2_widget)
        m2_layout.setContentsMargins(0, 0, 0, 0)

        self.m2_table = QTableWidget(0, 7)
        self.m2_table.setHorizontalHeaderLabels(["Z(mm)", "D4σ X", "D4σ Y", "1/e² X", "1/e² Y", "刀口 X", "刀口 Y"])
        self.m2_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.m2_table.verticalHeader().setDefaultSectionSize(35)
        self.m2_table.verticalHeader().setVisible(False)
        m2_layout.addWidget(self.m2_table)

        self.right_stacked.addWidget(self.right_normal_widget)
        self.right_stacked.addWidget(self.right_m2_widget)
        right_layout.addWidget(self.right_stacked)

        workspace_layout.addWidget(self.right_panel, stretch=1)
        main_layout.addLayout(workspace_layout)

        size_grip = QSizeGrip(self)
        main_layout.addWidget(size_grip, 0, Qt.AlignBottom | Qt.AlignRight)

        # 保持外侧大方框为白色实线
        self.pen_outline = pg.mkPen(color='#FFFFFF', width=4.0, style=Qt.SolidLine)
        self.beam_ring_outline = pg.PlotCurveItem(pen=self.pen_outline)
        self.beam_ring_outline.setZValue(14)
        self.plot_widget.addItem(self.beam_ring_outline)

        self.beam_ring.setPen(self.pen_ring)
        self.beam_ring.setZValue(15)

    def toggle_phase_window(self):
        if self.btn_phase_start.isChecked():
            if not hasattr(self, 'phase_window') or self.phase_window is None:
                self.phase_window = PhaseReconstructionWindow(self)

            geom = self.geometry()
            self.phase_window.move(geom.right() + 10, geom.top())
            self.phase_window.show()

            self.change_colormap()
            self.btn_phase_start.setText(self.tr("Stop Reconstruction", "停止重建"))
        else:
            if hasattr(self, 'phase_window') and self.phase_window:
                self.phase_window.hide()
            self.btn_phase_start.setText(self.tr("Start Reconstruction", "开始重建"))

    def sync_phase_layout(self):
        if hasattr(self, 'phase_window') and self.phase_window and self.phase_window.isVisible():
            self.phase_window.update_layout()

    # ================= M² 功能专属交互 =================
    def toggle_m2_mode(self):
        if self.btn_m2_mode.isChecked():
            self.btn_m2_mode.setText(self.tr("Exit M² Measurement", "退出 M² 测量"))
            self.right_stacked.setCurrentIndex(1)
            self.m2_table.setRowCount(0)
            self.m2_data_list = []
        else:
            self.btn_m2_mode.setText(self.tr("Start M² Measurement", "开启 M² 测量"))
            self.right_stacked.setCurrentIndex(0)

    def record_m2_data(self):
        if not self.btn_m2_mode.isChecked(): return
        if self.m2_table.rowCount() >= 20:
            QMessageBox.warning(self, "上限", "最多只能记录 20 行数据！")
            return
        if self.current_sig_x == 0 or self.current_sig_y == 0:
            QMessageBox.warning(self, "无信号", "当前未能检测到有效光斑，无法提取束腰宽度！")
            return

        z_pos, ok = QInputDialog.getDouble(self, "输入位置", "请输入当前相机Z轴位置 (mm):", 0, -10000, 10000, 2)
        if not ok: return

        d4s_x = 4.0 * self.current_sig_x * self.pixel_size_um
        d4s_y = 4.0 * self.current_sig_y * self.pixel_size_um
        e2_x = 2.828 * self.current_sig_x * self.pixel_size_um
        e2_y = 2.828 * self.current_sig_y * self.pixel_size_um
        knife_x = 3.2 * self.current_sig_x * self.pixel_size_um
        knife_y = 3.2 * self.current_sig_y * self.pixel_size_um

        data = {
            'z': z_pos,
            'd4s_x': d4s_x, 'd4s_y': d4s_y,
            'e2_x': e2_x, 'e2_y': e2_y,
            'knife_x': knife_x, 'knife_y': knife_y
        }
        self.m2_data_list.append(data)

        row = self.m2_table.rowCount()
        self.m2_table.insertRow(row)
        self.m2_table.setItem(row, 0, QTableWidgetItem(f"{z_pos:.2f}"))
        self.m2_table.setItem(row, 1, QTableWidgetItem(f"{d4s_x:.1f}"))
        self.m2_table.setItem(row, 2, QTableWidgetItem(f"{d4s_y:.1f}"))
        self.m2_table.setItem(row, 3, QTableWidgetItem(f"{e2_x:.1f}"))
        self.m2_table.setItem(row, 4, QTableWidgetItem(f"{e2_y:.1f}"))
        self.m2_table.setItem(row, 5, QTableWidgetItem(f"{knife_x:.1f}"))
        self.m2_table.setItem(row, 6, QTableWidgetItem(f"{knife_y:.1f}"))
        self.m2_table.scrollToBottom()

    def delete_m2_data(self):
        if not self.m2_data_list: return
        self.m2_data_list.pop()
        self.m2_table.removeRow(self.m2_table.rowCount() - 1)

    def calc_plot_m2(self):
        if len(self.m2_data_list) < 3:
            QMessageBox.warning(self, "数据不足", "拟合抛物线至少需要记录 3 个不同位置的数据！")
            return

        algorithms = [
            ("D4σ", 'd4s_x', 'd4s_y'),
            ("1/e²", 'e2_x', 'e2_y'),
            ("刀口法 (Knife Edge)", 'knife_x', 'knife_y')
        ]

        z_arr = np.array([d['z'] for d in self.m2_data_list])
        lambda_mm = self.spin_wave.value() / 1e6

        self.m2_plot_windows = []
        for title, key_x, key_y in algorithms:
            d_x_um = [d[key_x] for d in self.m2_data_list]
            d_y_um = [d[key_y] for d in self.m2_data_list]
            dialog = M2PlotWindow(title, z_arr, d_x_um, d_y_um, lambda_mm, parent=self)
            dialog.show()
            self.m2_plot_windows.append(dialog)

    # ================= 主题引擎 =================
    def switch_theme(self, index):
        self.current_theme = 'dark' if index == 0 else 'light'
        self.apply_theme()

    def apply_theme(self):
        if self.current_theme == 'light':
            bg_main = "#F5F5F5"
            bg_panel = "#FFFFFF"
            bg_tab = "#EFEFEF"
            bg_btn = "#E0E0E0"
            bg_btn_hover = "#D0D0D0"
            bg_input = "#FFFFFF"
            border_color = "#CCCCCC"
            text_color = "#333333"
            text_accent = "#005A9E"
            plot_bg = "#FFFFFF"
            profile_pen = pg.mkPen('k', width=1)
            val_dev_color = "#008800"
            val_calc_color = "#005A9E"
            val_bg = "#F9F9F9"
            chk_border = "#777777"
            chk_checked_bg = "#005A9E"
        else:
            bg_main = "#222222"
            bg_panel = "#333333"
            bg_tab = "#2B2B2B"
            bg_btn = "#4A4A4A"
            bg_btn_hover = "#5A5A5A"
            bg_input = "#4A4A4A"
            border_color = "#555555"
            text_color = "#E0E0E0"
            text_accent = "#00A2FF"
            plot_bg = "#111111"
            profile_pen = pg.mkPen('w', width=1)
            val_dev_color = "#00FF00"
            val_calc_color = "#00A2FF"
            val_bg = "#222222"
            chk_border = "#AAAAAA"
            chk_checked_bg = "#00A2FF"

        theme_qss = f"""
            QMainWindow, QDialog {{ background-color: {bg_main}; border: 1px solid {border_color}; }}
            QWidget {{ color: {text_color}; font-size: 16px; font-weight: bold; font-family: "Segoe UI", "Microsoft YaHei", sans-serif; }}
            QTabWidget::pane {{ border: 1px solid {border_color}; background: {bg_tab}; top: -1px; }}

            QTabBar::tab {{ background: {bg_main}; border: 1px solid {border_color}; font-size: 20px; padding: 10px 30px; min-width: 150px; margin-right: 4px; border-top-left-radius: 6px; border-top-right-radius: 6px;}}
            QTabBar::tab:selected {{ background: {bg_tab}; border-bottom-color: {bg_tab}; color: {text_accent};}}
            QTabBar::tab:hover {{ background: {bg_panel}; }}

            QGroupBox {{ border: 1px solid {border_color}; margin-top: 28px; padding-top: 14px; border-radius: 4px; background-color: {bg_panel};}}
            QGroupBox::title {{ subcontrol-origin: margin; subcontrol-position: top left; left: 15px; top: 4px; padding: 0 5px; color: {text_accent};}}

            QPushButton {{ background-color: {bg_btn}; color: {text_color}; border: 1px solid {border_color}; padding: 8px 20px; border-radius: 5px; font-size: 18px; }}
            QPushButton:hover {{ background-color: {bg_btn_hover}; border: 1px solid {text_accent}; }}
            QPushButton:pressed {{ background-color: {text_accent}; color: white; border: 1px solid {text_accent}; }}
            QPushButton:checked {{ background-color: {text_accent}; color: white; border: 1px solid {text_accent}; }}
            QPushButton:checked:hover {{ background-color: #0078D7; }}
            QPushButton:disabled {{ background-color: {bg_panel}; color: #777777; border: 1px solid {border_color}; }}

            QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{ background-color: {bg_input}; color: {text_color}; border: 1px solid {border_color}; padding: 6px; border-radius: 4px; min-width: 100px;}}
            QComboBox QAbstractItemView {{ background-color: {bg_input}; color: {text_color}; selection-background-color: {text_accent}; }}

            QCheckBox, QRadioButton {{ spacing: 10px; }}
            QCheckBox::indicator, QRadioButton::indicator {{ width: 16px; height: 16px; background-color: transparent; border: 2px solid {chk_border}; border-radius: 2px;}}
            QCheckBox::indicator:hover, QRadioButton::indicator:hover {{ border: 2px solid {text_accent}; }}
            QCheckBox::indicator:checked, QRadioButton::indicator:checked {{ background-color: {chk_checked_bg}; border: 2px solid {text_accent}; }}
            QRadioButton::indicator {{ border-radius: 8px; }}

            QTableWidget {{ background-color: {bg_input}; color: {text_color}; gridline-color: {border_color}; border: none; }}
            QHeaderView::section {{ background-color: {bg_panel}; color: {text_color}; border: 1px solid {border_color}; padding: 6px; }}
        """
        self.setStyleSheet(theme_qss)
        self.top_widget.setStyleSheet(f"background-color: {bg_main};")
        self.content_widget.setStyleSheet(f"background-color: {bg_main};")
        self.title_bar.apply_theme(self.current_theme)

        self.plot_widget.setBackground(plot_bg)
        self.curve_x.setPen(profile_pen)
        self.curve_y.setPen(profile_pen)

        dev_style = f"color: {val_dev_color}; background: {val_bg}; padding: 4px; border: 1px solid {border_color};"
        calc_style = f"color: {val_calc_color}; background: {val_bg}; padding: 4px; border: 1px solid {border_color};"

        for lbl in self.dev_val_labels:
            lbl.setFont(self.val_font)
            lbl.setStyleSheet(dev_style)
        for lbl in self.calc_val_labels:
            lbl.setFont(self.val_font)
            lbl.setStyleSheet(calc_style)
        self.update_status_label()

    def update_status_label(self):
        border_color = "#CCCCCC" if self.current_theme == 'light' else "#444444"
        bg_color = "#F9F9F9" if self.current_theme == 'light' else "#222222"
        base_style = f"background: {bg_color}; padding: 4px; border: 1px solid {border_color};"

        if self.btn_sim_mode.isChecked():
            self.lbl_conn_status.setText(self.tr("Simulating 🟢", "虚拟连接🟢"))
            conn_color = "#00A2FF"
        elif not HAS_MVSDK:
            self.lbl_conn_status.setText(self.tr("No SDK ❌", "环境缺失❌"))
            conn_color = "#777777"
        elif self.is_connected:
            self.lbl_conn_status.setText(self.tr("Connected 🟢", "连接成功🟢"))
            conn_color = "#008800" if self.current_theme == 'light' else "#00FF00"
        else:
            if hasattr(self, 'connection_failed') and self.connection_failed:
                self.lbl_conn_status.setText(self.tr("Connect Failed 🔴", "连接失败🔴"))
                conn_color = "#E81123"
            else:
                self.lbl_conn_status.setText(self.tr("No Device ❌", "无连接设备❌"))
                conn_color = "#D26900" if self.current_theme == 'light' else "#FFA500"

        self.lbl_conn_status.setStyleSheet(f"color: {conn_color}; {base_style}")

        if not self.is_connected and not self.btn_sim_mode.isChecked():
            self.lbl_cap_status.setText(self.tr("Standby", "未就绪"))
            cap_color = "#777777"
        elif self.is_running:
            if self.is_paused:
                self.lbl_cap_status.setText(self.tr("Paused ⏸️", "暂停⏸️"))
                cap_color = "#D26900" if self.current_theme == 'light' else "#FFA500"
            else:
                self.lbl_cap_status.setText(self.tr("Capturing ⏯️", "采集中⏯️"))
                cap_color = "#008800" if self.current_theme == 'light' else "#00FF00"
        else:
            self.lbl_cap_status.setText(self.tr("Stopped", "停止"))
            cap_color = "#D26900" if self.current_theme == 'light' else "#FFA500"

        self.lbl_cap_status.setStyleSheet(f"color: {cap_color}; {base_style}")

    def open_capture_dialog(self):
        dlg = CaptureDialog(self)
        dlg.exec_()

    def open_save_dialog(self):
        dlg = SaveDialog(self)
        dlg.exec_()

    def switch_language(self, index):
        self.current_lang = 'zh' if index == 0 else 'en'
        self.retranslate_ui()

    def retranslate_ui(self):
        self.title_bar.title_label.setText(self.tr("Beam Profiler - Professional Edition", "激光光束分析仪 - 专业版"))
        self.ribbon.setTabText(0, self.tr("Capture", "采集"))
        self.ribbon.setTabText(1, self.tr("Display", "显示"))
        self.ribbon.setTabText(2, self.tr("Computations", "计算"))
        self.ribbon.setTabText(3, self.tr("Beam Quality", "光束质量"))
        self.ribbon.setTabText(4, self.tr("About", "关于"))

        self.grp_device_mgr.setTitle(self.tr("Device Management", "设备管理"))
        self.btn_scan.setText(self.tr("Scan Devices", "扫描设备"))

        if self.is_connected:
            self.btn_connect.setText(self.tr("Disconnect 🔴", "连接断开🔴"))
        else:
            self.btn_connect.setText(self.tr("Connect Device", "设备连接"))

        self.grp_control.setTitle(self.tr("Control", "控制"))
        self.btn_pause.setText(self.tr("Resume", "恢复") if self.is_paused else self.tr("Pause", "暂停"))

        self.grp_exposure.setTitle(self.tr("Exposure & Gain", "曝光与增益"))
        self.chk_auto_exp.setText(self.tr("Auto", "自动"))
        self.lbl_exp.setText(self.tr("Exp:", "曝光:"))
        self.lbl_gain.setText(self.tr("Gain:", "增益:"))
        self.grp_actions.setTitle(self.tr("Actions", "附加操作"))
        self.btn_capture.setText(self.tr("Capture", "采集单帧"))
        self.btn_save.setText(self.tr("Save", "保存数据"))
        self.btn_bg_sub.setText(self.tr("Subtract BG", "去除底噪"))
        self.grp_color.setTitle(self.tr("Color Palette", "色彩映射"))
        self.lbl_cmap.setText(self.tr("Colormap:", "当前色谱:"))

        current_idx = self.combo_colormap.currentIndex() if self.combo_colormap.count() > 0 else 0
        self.combo_colormap.blockSignals(True)
        self.combo_colormap.clear()

        self.combo_colormap.addItem(self.tr("Rainbow", "Rainbow"), "rainbow")
        self.combo_colormap.addItem(self.tr("Jet", "Jet"), "jet")
        self.combo_colormap.addItem(self.tr("Dark Blue", "DarkBlue"), "darkblue")
        self.combo_colormap.addItem(self.tr("Green", "Green"), "green")
        self.combo_colormap.addItem(self.tr("Red", "Red"), "red")
        self.combo_colormap.addItem(self.tr("Violet", "Violet"), "violet")
        self.combo_colormap.addItem(self.tr("Parula (MATLAB)", "Parula (MATLAB默认)"), "parula")
        self.combo_colormap.addItem(self.tr("Viridis (Scientific)", "Viridis (科学护眼)"), "viridis")
        self.combo_colormap.addItem(self.tr("HSV", "HSV (高饱和度)"), "hsv")
        self.combo_colormap.addItem(self.tr("Hot", "Hot (标准热图)"), "hot")
        self.combo_colormap.addItem(self.tr("Bone", "Bone (骨色灰图)"), "bone")

        self.combo_colormap.setCurrentIndex(current_idx)
        self.combo_colormap.blockSignals(False)
        self.change_colormap()

        self.grp_2d.setTitle(self.tr("2D Overlays", "2D 图层"))
        self.chk_crosshair.setText(self.tr("Crosshair", "十字光标"))
        self.chk_profiles.setText(self.tr("Profiles", "边缘曲线"))
        self.grp_roi.setTitle(self.tr("Aperture Shapes", "手动光阑形状"))

        self.btn_circ.setText(self.tr("Circle", "圆形"))
        self.btn_sq.setText(self.tr("Square", "正方形"))
        self.btn_rect.setText(self.tr("Rectangle", "长方形"))
        self.btn_ell.setText(self.tr("Ellipse", "椭圆形"))
        self.grp_calc.setTitle(self.tr("Fitting Algorithms", "高级拟合算法"))
        self.chk_knife.setText(self.tr("Knife Edge", "刀口法"))

        self.grp_phase.setTitle(self.tr("Phase Retrieval", "相位重建模块"))
        if hasattr(self, 'phase_window') and self.phase_window and self.phase_window.isVisible():
            self.btn_phase_start.setText(self.tr("Stop Reconstruction", "停止重建"))
        else:
            self.btn_phase_start.setText(self.tr("Start Reconstruction", "开始重建"))
        self.chk_phase_fringes.setText(self.tr("Fringes", "干涉条纹"))

        self.grp_m2.setTitle(self.tr("M² Measurement Configuration", "M² 测量与光束拟合参数配置"))
        if self.btn_m2_mode.isChecked():
            self.btn_m2_mode.setText(self.tr("Exit M² Measurement", "退出 M² 测量"))
        else:
            self.btn_m2_mode.setText(self.tr("Start M² Measurement", "开启 M² 测量"))
        self.lbl_wave.setText(self.tr("Wavelength (nm):", "波长 (nm):"))
        self.lbl_focal.setText(self.tr("Lens Focal(mm):", "透镜焦距(mm):"))
        self.btn_focal_confirm.setText(self.tr("Confirm", "确定"))
        self.btn_m2_record.setText(self.tr("Record Data", "记录数据"))
        self.btn_m2_delete.setText(self.tr("Delete Last Data", "删除上次数据"))
        self.btn_m2_calc.setText(self.tr("Calculate & Plot M²", "拟合计算光束质量"))

        self.grp_sim.setTitle(self.tr("Simulation", "仿真建模"))
        self.btn_sim_mode.setText(self.tr("Enable Sim", "开启仿真"))
        self.radio_gauss.setText(self.tr("Gaussian", "高斯光束"))
        self.radio_sgauss.setText(self.tr("Super-Gaussian", "超高斯光束"))
        self.radio_hgauss.setText(self.tr("Hermite-Gaussian", "厄米高斯光束"))
        self.radio_lgauss.setText(self.tr("Laguerre-Gaussian", "拉盖尔高斯"))

        if hasattr(self, 'chk_sim_jitter'):
            self.chk_sim_jitter.setText(self.tr("Beam Jitter", "光束指向晃动"))
        if hasattr(self, 'chk_sim_interf'):
            self.chk_sim_interf.setText(self.tr("Interference", "平面波干涉"))

        self.grp_lang.setTitle(self.tr("Language", "语言设置"))
        self.grp_theme.setTitle(self.tr("Theme", "主题风格"))

        current_theme_idx = self.combo_theme.currentIndex()
        self.combo_theme.blockSignals(True)
        self.combo_theme.clear()
        self.combo_theme.addItem(self.tr("Dark Theme", "深色模式 (Dark)"), "dark")
        self.combo_theme.addItem(self.tr("Light Theme", "浅色模式 (Light)"), "light")
        self.combo_theme.setCurrentIndex(current_theme_idx)
        self.combo_theme.blockSignals(False)

        self.grp_info.setTitle(self.tr("Software Info", "软件信息"))
        self.lbl_app_name.setText(self.tr("Beam Profiler App", "智能激光光束分析系统"))

        self.grp_device_params.setTitle(self.tr("Device Parameters", "设备参数"))
        self.lbl_t_model.setText(self.tr("Model:", "相机型号:"))
        self.lbl_t_conn_status.setText(self.tr("Connection:", "连接状态:"))
        self.lbl_t_cap_status.setText(self.tr("Capture:", "采集状态:"))
        self.lbl_t_fps.setText(self.tr("FPS:", "采集帧率:"))
        self.lbl_t_exp_res.setText(self.tr("Exposure:", "实际曝光:"))
        self.lbl_t_gain_res.setText(self.tr("Gain:", "实际增益:"))
        self.lbl_t_noise.setText(self.tr("Noise (dB):", "背景噪声:"))

        self.update_status_label()

        self.grp_calc_params.setTitle(self.tr("Calculation", "计算参数"))
        self.lbl_t_peak.setText(self.tr("Peak Val:", "峰值强度:"))
        self.lbl_t_centroid.setText(self.tr("Centroid:", "能量质心:"))
        self.lbl_t_width_x.setText(self.tr("X Width:", "X轴宽度:"))
        self.lbl_t_width_y.setText(self.tr("Y Width:", "Y轴宽度:"))
        self.lbl_t_aperture.setText(self.tr("Aperture:", "光阑尺寸:"))

    # ================= 交互控制操作 =================

    def change_colormap(self):
        map_key = self.combo_colormap.currentData()
        if map_key and map_key in self.colormaps_data:
            cmap = self.colormaps_data[map_key]
            self.image_item.setColorMap(cmap)

            if hasattr(self, 'phase_window') and self.phase_window and self.phase_window.isVisible():
                self.phase_window.sync_colormap(cmap, self._colorbar_buffer_qimg(cmap))

            lut = cmap.getLookupTable(0.0, 1.0, 256).astype(np.uint8)
            lut = lut[::-1]

            channels = lut.shape[1]

            bar_img = np.zeros((256, 1, channels), dtype=np.uint8)
            bar_img[:, 0, :] = lut

            self._colorbar_buffer = np.ascontiguousarray(bar_img)

            img_format = QImage.Format_RGBA8888 if channels == 4 else QImage.Format_RGB888
            qimg = QImage(self._colorbar_buffer.data, 1, 256, channels, img_format)

            self.colorbar_label.setPixmap(QPixmap.fromImage(qimg))
            self.colorbar_label.setScaledContents(True)
            self.colorbar_label.setStyleSheet("border: 1px solid #555;")

    def _colorbar_buffer_qimg(self, cmap):
        lut = cmap.getLookupTable(0.0, 1.0, 256).astype(np.uint8)
        lut = lut[::-1]
        channels = lut.shape[1]
        bar_img = np.zeros((256, 1, channels), dtype=np.uint8)
        bar_img[:, 0, :] = lut
        bar_img = np.ascontiguousarray(bar_img)
        img_format = QImage.Format_RGBA8888 if channels == 4 else QImage.Format_RGB888
        return QImage(bar_img.data, 1, 256, channels, img_format)

    def toggle_crosshair(self, state):
        visible = (state == Qt.Checked)
        self.v_line.setVisible(visible)
        self.h_line.setVisible(visible)

    # ★ 彻底解耦：普通边缘曲线完全独立
    def toggle_profiles(self, state):
        visible = (state == Qt.Checked)
        self.curve_x.setVisible(visible)
        self.curve_y.setVisible(visible)

    # ★ 彻底解耦：拟合相关图形由拟合算法选项独立控制
    def toggle_fit_method(self, clicked_chk):
        for chk in [self.chk_d4s, self.chk_1e2, self.chk_knife]:
            if chk != clicked_chk:
                chk.setChecked(False)

        is_fitting = self.chk_d4s.isChecked() or self.chk_1e2.isChecked() or self.chk_knife.isChecked()
        if not is_fitting:
            self.fit_curve_x.hide()
            self.fit_curve_y.hide()

    def on_roi_changed(self):
        if self.current_roi_type:
            roi_size = self.rois[self.current_roi_type].size()
            w_um = roi_size[0] * self.pixel_size_um
            h_um = roi_size[1] * self.pixel_size_um
            self.lbl_aperture.setText(f"{w_um:.1f} x {h_um:.1f} um")
        else:
            self.lbl_aperture.setText("N/A")

    def toggle_aperture(self, shape_type):
        btn_map = {
            'circle': self.btn_circ,
            'square': self.btn_sq,
            'rect': self.btn_rect,
            'ellipse': self.btn_ell
        }
        if self.current_roi_type == shape_type:
            self.rois[shape_type].hide()
            btn_map[shape_type].setChecked(False)
            self.current_roi_type = None
            self.on_roi_changed()
            return
        if self.current_roi_type:
            self.rois[self.current_roi_type].hide()
            btn_map[self.current_roi_type].setChecked(False)

        self.current_roi_type = shape_type
        btn_map[shape_type].setChecked(True)

        view_rect = self.plot_widget.getViewBox().viewRect()
        center_x = view_rect.center().x()
        center_y = view_rect.center().y()
        size = min(view_rect.width(), view_rect.height()) * 0.4

        self.rois[shape_type].setPos([center_x - size / 2, center_y - size / 2])
        self.rois[shape_type].setSize([size, size])
        self.rois[shape_type].show()
        self.on_roi_changed()

    def on_slider_exp_changed(self, val):
        if self.chk_auto_exp.isChecked(): return
        ms_val = 0.0049 + (val / 10000.0)
        self.spin_exp.blockSignals(True)
        self.spin_exp.setValue(ms_val)
        self.spin_exp.blockSignals(False)
        self.apply_exposure_hardware()

    def on_spin_exp_changed(self, val):
        if self.chk_auto_exp.isChecked(): return
        slider_val = int(round((val - 0.0049) * 10000.0))
        self.slider_exp.blockSignals(True)
        self.slider_exp.setValue(slider_val)
        self.slider_exp.blockSignals(False)
        self.apply_exposure_hardware()

    def apply_exposure_hardware(self):
        ms_val = self.spin_exp.value()
        self.lbl_exp_res.setText(f"{ms_val:.4f} ms")
        if HAS_MVSDK and self.h_camera:
            try:
                mvdll.CameraSetExposureTime(self.h_camera, ctypes.c_double(float(ms_val * 1000.0)))
            except:
                pass

    def on_gain_changed(self):
        self.lbl_gain_res.setText(f"{self.slider_gain.value()} %")
        if HAS_MVSDK and self.h_camera:
            try:
                mvdll.CameraSetAnalogGain(self.h_camera, self.slider_gain.value())
            except:
                pass

    def on_auto_exp_changed(self, state):
        is_auto = (state == Qt.Checked)
        self.slider_exp.setEnabled(not is_auto)
        self.spin_exp.setEnabled(not is_auto)
        if HAS_MVSDK and self.h_camera:
            try:
                mvdll.CameraSetAeState(self.h_camera, 1 if is_auto else 0)
                if not is_auto:
                    self.apply_exposure_hardware()
            except:
                pass

    def scan_devices(self):
        if self.btn_sim_mode.isChecked():
            self.btn_sim_mode.setChecked(False)
            self.on_sim_mode_toggled()

        if not HAS_MVSDK:
            QMessageBox.critical(self, "错误", "底层库缺失，无法扫描设备。")
            return

        devs = (tSdkCameraDevInfo * 16)()
        count = ctypes.c_int(16)
        try:
            mvdll.CameraEnumerateDevice(devs, ctypes.byref(count))
        except Exception as e:
            QMessageBox.critical(self, "异常", f"扫描异常: {e}")
            return

        self.device_list = []
        self.combo_devices.blockSignals(True)
        self.combo_devices.clear()

        num_devs = count.value
        if num_devs == 0:
            self.combo_devices.addItem("未发现可用设备")
        else:
            for i in range(num_devs):
                dev_info = devs[i]
                name = dev_info.acFriendlyName.decode('gbk', 'ignore')
                self.device_list.append(dev_info)
                self.combo_devices.addItem(name)

        self.combo_devices.blockSignals(False)

    def connect_device(self):
        if self.btn_sim_mode.isChecked():
            self.btn_sim_mode.setChecked(False)
            self.on_sim_mode_toggled()

        if not HAS_MVSDK or not self.device_list:
            QMessageBox.warning(self, "提示", "请先扫描并确保有可用相机！")
            return

        if self.is_connected:
            self.btn_connect.setEnabled(False)
            self.stop_real_camera()
            if self.h_camera is not None:
                try:
                    mvdll.CameraUnInit(self.h_camera)
                except:
                    pass
                self.h_camera = None
            self.is_connected = False
            self.is_running = False
            self.connection_failed = False
            self.btn_connect.setText(self.tr("Connect Device", "设备连接"))
            self.btn_connect.setStyleSheet("")
            self.btn_connect.setEnabled(True)
            self.update_status_label()
            self.clear_display_data()
            return

        idx = self.combo_devices.currentIndex()
        if idx < 0 or idx >= len(self.device_list): return

        self.btn_connect.setEnabled(False)

        dev_info = self.device_list[idx]
        self.h_camera = ctypes.c_int()
        res = mvdll.CameraInit(ctypes.byref(dev_info), -1, -1, ctypes.byref(self.h_camera))

        if res != 0:
            QMessageBox.critical(self, "连接失败", f"初始化设备失败，错误码: {res}")
            self.connection_failed = True
            self.is_connected = False
            self.btn_connect.setEnabled(True)
            self.update_status_label()
            return

        self.connection_failed = False
        self.h_camera = self.h_camera.value
        CAMERA_MEDIA_TYPE_MONO8 = 0x01000000 | 0x00080000 | 0x0001
        mvdll.CameraSetIspOutFormat(self.h_camera, CAMERA_MEDIA_TYPE_MONO8)
        mvdll.CameraSetTriggerMode(self.h_camera, 0)

        self.is_connected = True
        self.lbl_model.setText(dev_info.acFriendlyName.decode('gbk', 'ignore'))

        self.start_real_camera()

        self.flash_count = 0
        self.flash_timer = QTimer()
        self.flash_timer.timeout.connect(self._flash_connect_btn)
        self.flash_timer.start(150)

        self.update_status_label()

    def _flash_connect_btn(self):
        self.flash_count += 1
        if self.flash_count % 2 == 1:
            self.btn_connect.setStyleSheet("background-color: #00FF00; color: black; border-color: #00FF00;")
        else:
            self.btn_connect.setStyleSheet("")

        if self.flash_count >= 6:
            self.flash_timer.stop()
            self.btn_connect.setStyleSheet("")
            self.btn_connect.setText(self.tr("Disconnect 🔴", "连接断开🔴"))
            self.btn_connect.setEnabled(True)

    def start_real_camera(self):
        if not self.is_connected or self.h_camera is None: return
        try:
            is_auto = 1 if self.chk_auto_exp.isChecked() else 0
            mvdll.CameraSetAeState(self.h_camera, is_auto)
            if not is_auto:
                self.apply_exposure_hardware()
            mvdll.CameraSetAnalogGain(self.h_camera, self.slider_gain.value())

            mvdll.CameraPlay(self.h_camera)

            self.camera_data = {
                'hCamera': self.h_camera,
                'buffer_size': 5000 * 5000 * 3,
                'mono': True,
                'opened': True
            }

            self.camera_thread = CameraAcquisitionThread(self, self.camera_data)
            self.camera_thread.start()

            self.force_auto_range = True
            self.ui_render_timer.start(16)

            self.is_running = True
            self.update_status_label()

        except Exception as e:
            QMessageBox.critical(self, "硬件异常", f"启动采集异常:\n{str(e)}")

    def stop_real_camera(self):
        self.ui_render_timer.stop()
        if self.camera_thread and self.camera_thread.isRunning():
            self.camera_thread.stop()

        if self.h_camera is not None:
            try:
                mvdll.CameraPause(self.h_camera)
            except:
                pass

        self.is_running = False
        self.update_status_label()

    def toggle_pause(self):
        if not self.is_running and not self.is_paused:
            self.btn_pause.setChecked(False)
            return

        if self.btn_pause.isChecked():
            self.btn_pause.setText(self.tr("Resume", "恢复"))
            self.is_paused = True
        else:
            self.btn_pause.setText(self.tr("Pause", "暂停"))
            self.is_paused = False
        self.update_status_label()

    def on_sim_mode_toggled(self):
        if self.btn_sim_mode.isChecked():
            if self.is_running and not self.is_paused:
                self.stop_real_camera()
                if self.is_connected and self.h_camera is not None:
                    try:
                        mvdll.CameraUnInit(self.h_camera)
                    except:
                        pass
                    self.h_camera = None
                    self.is_connected = False
                    self.btn_connect.setText(self.tr("Connect Device", "设备连接"))
                    self.btn_connect.setStyleSheet("")
                    self.btn_connect.setEnabled(True)
            self.lbl_model.setText("Simulation Core")
            self.is_running = True
            self.update_status_label()

            self.force_auto_range = True

            self.sim_timer.start(10)
            self.ui_render_timer.start(16)
        else:
            self.sim_timer.stop()
            self.ui_render_timer.stop()
            self.is_running = False
            self.clear_display_data()
            self.update_status_label()

    def clear_display_data(self):
        self.latest_frame = None
        self.current_frame = None
        self.current_fps = 0.0
        self.sim_ideal_profile_x = None
        self.sim_ideal_profile_y = None
        self.image_item.clear()
        self.curve_x.setData([], [])
        self.curve_y.setData([], [])
        self.fit_curve_x.setData([], [])
        self.fit_curve_y.setData([], [])
        self.beam_ring.setData([], [])
        self.beam_ring.hide()

        if hasattr(self, 'beam_ring_outline'):
            self.beam_ring_outline.setData([], [])
            self.beam_ring_outline.hide()

        self.v_line.clear()
        self.h_line.clear()

        self.lbl_peak.setText("0")
        self.lbl_centroid.setText("(0.0, 0.0)")
        self.lbl_width_x.setText("0.0 um")
        self.lbl_width_y.setText("0.0 um")
        self.lbl_fps.setText("0.0")
        self.lbl_noise.setText("0.0 dB")

        if HAS_MVSDK and self.is_connected:
            idx = self.combo_devices.currentIndex()
            if idx >= 0:
                self.lbl_model.setText(self.device_list[idx].acFriendlyName.decode('gbk', 'ignore'))
        else:
            self.lbl_model.setText("MV Camera" if HAS_MVSDK else "No Device")

    def on_beam_type_changed(self):
        if not self.btn_sim_mode.isChecked(): return
        self.sim_params_widget.show()

        if self.radio_gauss.isChecked():
            self.lbl_param1.setText("Param 1:")
            self.spin_param1.setEnabled(False)
            self.lbl_param2.setText("Param 2:")
            self.spin_param2.setEnabled(False)
        elif self.radio_sgauss.isChecked():
            self.lbl_param1.setText("Order P:")
            self.spin_param1.setEnabled(True)
            self.spin_param1.setMinimum(2)
            self.lbl_param2.setText("-")
            self.spin_param2.setEnabled(False)
        elif self.radio_hgauss.isChecked():
            self.lbl_param1.setText("Order m (X):")
            self.spin_param1.setEnabled(True)
            self.spin_param1.setMinimum(0)
            self.lbl_param2.setText("Order n (Y):")
            self.spin_param2.setEnabled(True)
            self.spin_param2.setMinimum(0)
        elif self.radio_lgauss.isChecked():
            self.lbl_param1.setText("Radial p:")
            self.spin_param1.setEnabled(True)
            self.spin_param1.setMinimum(0)
            self.lbl_param2.setText("Azimuthal l:")
            self.spin_param2.setEnabled(True)
            self.spin_param2.setMinimum(0)

    # ================= 主循环 (修复解耦：无论仿真/相机，强开拟合侧边线) =================
    def render_process_loop(self):
        if self.latest_frame is None or self.is_paused:
            return

        raw_data = self.latest_frame.astype(np.float32)
        self.latest_frame = None
        self.current_frame = raw_data.astype(np.uint8)
        self.image_item.setImage(self.current_frame, autoLevels=False, levels=(0, 255), autoDownsample=True)

        if self.force_auto_range:
            self.plot_widget.getViewBox().autoRange(items=[self.image_item], padding=0)
            self.force_auto_range = False

        H, W = raw_data.shape
        peak_val = np.max(raw_data)
        x_indices = np.arange(W)
        y_indices = np.arange(H)

        is_d4s = self.chk_d4s.isChecked()
        is_1e2 = self.chk_1e2.isChecked()
        is_knife = self.chk_knife.isChecked()
        is_fitting = is_d4s or is_1e2 or is_knife

        dx, dy = 0, 0
        cx, cy = W / 2.0, H / 2.0

        edge_samples = np.concatenate([raw_data[:5, :].flatten(), raw_data[-5:, :].flatten(),
                                       raw_data[:, :5].flatten(), raw_data[:, -5:].flatten()])
        noise_floor = np.mean(edge_samples) + 2 * np.std(edge_samples)
        threshold = max(noise_floor, peak_val * 0.03)
        img_clean = np.clip(raw_data - threshold, 0, 255)
        total_p = np.sum(img_clean)

        if total_p > 0:
            cx = np.dot(x_indices, np.sum(img_clean, axis=0)) / total_p
            cy = np.dot(y_indices, np.sum(img_clean, axis=1)) / total_p

            if self.chk_profiles.isChecked():
                prof_x_data = img_clean[min(int(cy), H - 1), :]
                prof_y_data = img_clean[:, min(int(cx), W - 1)]
                p_x_norm = prof_x_data / (np.max(prof_x_data) if np.max(prof_x_data) > 0 else 1) * 255
                p_y_norm = prof_y_data / (np.max(prof_y_data) if np.max(prof_y_data) > 0 else 1) * 255
                self.curve_x.setData(x=x_indices, y=H - (p_x_norm / 255.0 * (H * 0.2)))
                self.curve_y.setData(x=(p_y_norm / 255.0 * (W * 0.2)), y=y_indices)
                self.curve_x.show()
                self.curve_y.show()
            else:
                self.curve_x.hide()
                self.curve_y.hide()

            if is_fitting:
                var_x_raw = np.dot((x_indices - cx) ** 2, np.sum(img_clean, axis=0)) / total_p
                raw_sig = np.sqrt(max(1, var_x_raw))
                mask_r = max(50, raw_sig * 2.5)
                yy, xx = np.ogrid[:H, :W]
                img_masked = np.where((xx - cx) ** 2 + (yy - cy) ** 2 < mask_r ** 2, img_clean, 0)
                sum_m = np.sum(img_masked)

                if sum_m > 0:
                    prof_x = np.sum(img_masked, axis=0)
                    prof_y = np.sum(img_masked, axis=1)
                    sig_x = np.sqrt(max(0.1, np.dot((x_indices - cx) ** 2, prof_x) / sum_m))
                    sig_y = np.sqrt(max(0.1, np.dot((y_indices - cy) ** 2, prof_y) / sum_m))

                    self.current_sig_x = sig_x
                    self.current_sig_y = sig_y

                    factor = 2.828 if is_1e2 else (3.2 if is_knife else 4.0)
                    dx, dy = factor * sig_x, factor * sig_y

                    self.lbl_peak.setText(f"{int(peak_val)}")
                    self.lbl_centroid.setText(f"({cx:.1f}, {cy:.1f})")
                    self.lbl_width_x.setText(f"{dx * self.pixel_size_um:.0f} um")
                    self.lbl_width_y.setText(f"{dy * self.pixel_size_um:.0f} um")

                    theta = np.linspace(0, 2 * np.pi, 100)

                    if total_p > 0:
                        self.beam_ring_outline.setData(
                            x=cx + (dx / 2.0) * np.cos(theta),
                            y=cy + (dy / 2.0) * np.sin(theta)
                        )
                        self.beam_ring.setData(
                            x=cx + (dx / 2.0) * np.cos(theta),
                            y=cy + (dy / 2.0) * np.sin(theta)
                        )
                        self.beam_ring_outline.show()
                        self.beam_ring.show()

                    # ★ 彻底解耦：只要处于拟合状态，立刻强制绘制标准化高斯拟合白线，与仿真/采集模式无关
                    w_fit_x = dx / 2.0
                    w_fit_y = dy / 2.0
                    if w_fit_x <= 0: w_fit_x = 1.0
                    if w_fit_y <= 0: w_fit_y = 1.0

                    fit_x = np.exp(-2 * (x_indices - cx) ** 2 / w_fit_x ** 2)
                    fit_y = np.exp(-2 * (y_indices - cy) ** 2 / w_fit_y ** 2)

                    fit_x = fit_x / (np.max(fit_x) if np.max(fit_x) > 0 else 1) * peak_val
                    fit_y = fit_y / (np.max(fit_y) if np.max(fit_y) > 0 else 1) * peak_val

                    # 强行独立绘制拟合曲线
                    self.fit_curve_x.setData(x=x_indices, y=H - (fit_x / 255.0 * (H * 0.2)))
                    self.fit_curve_y.setData(x=(fit_y / 255.0 * (W * 0.2)), y=y_indices)
                    self.fit_curve_x.show()
                    self.fit_curve_y.show()

                    # 相位计算流节流机制 (突破 8Hz 瓶颈限制)
                    if hasattr(self, 'phase_window') and self.phase_window and self.phase_window.isVisible():
                        if not hasattr(self, 'last_calc_time'): self.last_calc_time = 0
                        if time.time() - self.last_calc_time >= 0.033:
                            self.last_calc_time = time.time()
                            self.phase_window.process_frame(raw_data, cx, cy, dx, dy, is_fitting)
            else:
                self.clear_calc_display()

        else:
            self.curve_x.hide()
            self.curve_y.hide()
            self.clear_calc_display()

        if self.chk_crosshair.isChecked() and total_p > 0:
            rect = self.plot_widget.getViewBox().viewRect()
            y_min, y_max = rect.top(), rect.bottom()
            x_min, x_max = rect.left(), rect.right()

            if is_fitting and dx > 0 and dy > 0:
                gap_x, gap_y = dx / 2.0 + 2, dy / 2.0 + 2

                self.v_line.setData(
                    x=np.array([cx, cx, cx, cx], dtype=float),
                    y=np.array([y_min, cy - gap_y, cy + gap_y, y_max], dtype=float),
                    connect='pairs'
                )
                self.h_line.setData(
                    x=np.array([x_min, cx - gap_x, cx + gap_x, x_max], dtype=float),
                    y=np.array([cy, cy, cy, cy], dtype=float),
                    connect='pairs'
                )
            else:
                self.v_line.setData(
                    x=np.array([cx, cx], dtype=float),
                    y=np.array([y_min, y_max], dtype=float),
                    connect='all'
                )
                self.h_line.setData(
                    x=np.array([x_min, x_max], dtype=float),
                    y=np.array([cy, cy], dtype=float),
                    connect='all'
                )

            self.v_line.show()
            self.h_line.show()
        else:
            self.v_line.hide()
            self.h_line.hide()

        self.lbl_fps.setText(f"{self.current_fps:.1f}")
        self.lbl_exp_res.setText(f"{self.current_actual_exp:.4f} ms")
        self.lbl_gain_res.setText(f"{self.slider_gain.value()} %")
        self.lbl_noise.setText(f"{noise_floor:.1f} dB")

        if self.chk_auto_exp.isChecked():
            self.spin_exp.blockSignals(True)
            self.slider_exp.blockSignals(True)
            self.spin_exp.setValue(self.current_actual_exp)
            slider_val = int(round((self.current_actual_exp - 0.0049) * 10000.0))
            slider_val = max(0, min(slider_val, 101885951))
            self.slider_exp.setValue(slider_val)
            self.spin_exp.blockSignals(False)
            self.slider_exp.blockSignals(False)

    def clear_calc_display(self):
        self.beam_ring.hide()
        if hasattr(self, 'beam_ring_outline'):
            self.beam_ring_outline.hide()
        self.fit_curve_x.hide()
        self.fit_curve_y.hide()
        self.lbl_width_x.setText("0 um")
        self.lbl_width_y.setText("0 um")
        self.lbl_peak.setText("0")

    def update_sim_frame(self):
        if not self.btn_sim_mode.isChecked() or self.is_paused: return

        if self.chk_auto_exp.isChecked():
            self.current_actual_exp = 10.0 + np.sin(self.sim_phase) * 2.0
        else:
            self.current_actual_exp = self.spin_exp.value()

        H, W = 600, 800
        y, x = np.ogrid[:H, :W]

        self.sim_phase += 0.1

        if hasattr(self, 'chk_sim_jitter') and self.chk_sim_jitter.isChecked():
            cx = W / 2 + np.sin(self.sim_phase * 0.5) * 50
            cy = H / 2 + np.cos(self.sim_phase * 0.3) * 30
        else:
            cx = W / 2
            cy = H / 2

        w0_x = 80 + np.sin(self.sim_phase) * 5
        w0_y = 70 + np.cos(self.sim_phase) * 5
        r2 = (x - cx) ** 2 + (y - cy) ** 2

        m_charge = 1
        theta_angle = np.arctan2(y - cy, x - cx)

        if self.radio_sgauss.isChecked():
            n = self.spin_param1.value()
            beam = np.exp(-((r2 / w0_x ** 2) ** n))
        elif self.radio_hgauss.isChecked():
            m, n = self.spin_param1.value(), self.spin_param2.value()
            if HAS_SCIPY:
                Hx = eval_hermite(m, np.sqrt(2) * (x - cx) / w0_x)
                Hy = eval_hermite(n, np.sqrt(2) * (y - cy) / w0_y)
                beam = (Hx * Hy) ** 2 * np.exp(-2 * ((x - cx) ** 2 / w0_x ** 2 + (y - cy) ** 2 / w0_y ** 2))
            else:
                beam = np.exp(-2 * r2 / w0_x ** 2)
        elif self.radio_lgauss.isChecked():
            p, l = self.spin_param1.value(), self.spin_param2.value()
            m_charge = l
            rho2 = 2 * r2 / w0_x ** 2
            if HAS_SCIPY:
                Lpl = genlaguerre(p, l)(rho2)
                beam = (rho2 ** l) * (Lpl ** 2) * np.exp(-rho2)
            else:
                beam = np.exp(-2 * r2 / w0_x ** 2)
        else:
            beam = np.exp(-2 * ((x - cx) ** 2 / w0_x ** 2 + (y - cy) ** 2 / w0_y ** 2))

        b_max = np.max(beam)
        if b_max > 0: beam = beam / b_max

        # 完美平面波干涉
        if hasattr(self, 'chk_sim_interf') and self.chk_sim_interf.isChecked():
            phase_obj = m_charge * theta_angle
            obj_complex = np.sqrt(beam) * np.exp(1j * phase_obj)

            kx = 80 * (2 * np.pi / W)
            ky = 40 * (2 * np.pi / H)

            ref_beam = np.exp(-2 * ((x - cx) ** 2 / (w0_x * 1.5) ** 2 + (y - cy) ** 2 / (w0_y * 1.5) ** 2))
            ref_complex = np.sqrt(ref_beam) * np.exp(1j * (kx * x + ky * y))

            interference = np.abs(obj_complex + ref_complex) ** 2
            interference = interference - np.min(interference)
            interference = interference / (np.max(interference) + 1e-12)

            beam_to_display = 0.05 + 0.90 * interference
        else:
            beam_to_display = beam

        noise_std = 0.015
        noise = np.random.normal(0, noise_std, (H, W))
        gray_img = np.clip((beam_to_display + noise) * 255, 0, 255).astype(np.uint8)

        self.latest_frame = gray_img
        self.current_fps = 60.0 + np.random.uniform(-0.5, 0.5)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    pg.setConfigOptions(antialias=True)
    window = BeamGageApp()
    window.show()
    sys.exit(app.exec_())
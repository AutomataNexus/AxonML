//! V4L2 Backend — Linux Camera Capture via Video4Linux2
//!
//! Pure Rust V4L2 camera capture using `libc` ioctls and memory-mapped buffers.
//! Supports YUYV and MJPEG formats from USB cameras and CSI (Pi) cameras.
//!
//! Only compiled on Linux (`#[cfg(target_os = "linux")]`).
//!
//! @version 0.1.0

use super::{CaptureBackend, CaptureConfig, CaptureError, FrameBuffer, PixelFormat};
use std::fs;
use std::os::unix::io::AsRawFd;

// =============================================================================
// V4L2 Constants
// =============================================================================

// V4L2 pixel format FourCC codes
const V4L2_PIX_FMT_YUYV: u32 = 0x5659_5559; // 'YUYV'
const V4L2_PIX_FMT_MJPEG: u32 = 0x4745_504D; // 'MJPG'

// V4L2 buffer types
const V4L2_BUF_TYPE_VIDEO_CAPTURE: u32 = 1;

// V4L2 memory types
const V4L2_MEMORY_MMAP: u32 = 1;

// V4L2 ioctl request codes (platform-dependent)
// These are Linux-specific, computed from _IOWR macros
const VIDIOC_QUERYCAP: libc::c_ulong = 0x8068_5600;
const VIDIOC_S_FMT: libc::c_ulong = 0xC0CC_5605;
const VIDIOC_REQBUFS: libc::c_ulong = 0xC014_5608;
const VIDIOC_QUERYBUF: libc::c_ulong = 0xC044_5609;
const VIDIOC_QBUF: libc::c_ulong = 0xC044_560F;
const VIDIOC_DQBUF: libc::c_ulong = 0xC044_5611;
const VIDIOC_STREAMON: libc::c_ulong = 0x4004_5612;
const VIDIOC_STREAMOFF: libc::c_ulong = 0x4004_5613;

const NUM_BUFFERS: u32 = 4;

// =============================================================================
// V4L2 Structures (C-compatible)
// =============================================================================

#[repr(C)]
struct V4l2Capability {
    driver: [u8; 16],
    card: [u8; 32],
    bus_info: [u8; 32],
    version: u32,
    capabilities: u32,
    device_caps: u32,
    reserved: [u32; 3],
}

#[repr(C)]
struct V4l2PixFormat {
    width: u32,
    height: u32,
    pixelformat: u32,
    field: u32,
    bytesperline: u32,
    sizeimage: u32,
    colorspace: u32,
    priv_: u32,
    flags: u32,
    ycbcr_enc: u32,
    quantization: u32,
    xfer_func: u32,
}

#[repr(C)]
struct V4l2Format {
    type_: u32,
    fmt: V4l2PixFormat,
    padding: [u8; 128], // Union padding
}

#[repr(C)]
struct V4l2RequestBuffers {
    count: u32,
    type_: u32,
    memory: u32,
    capabilities: u32,
    flags: u8,
    reserved: [u8; 3],
}

#[repr(C)]
struct V4l2Timeval {
    tv_sec: i64,
    tv_usec: i64,
}

#[repr(C)]
struct V4l2Buffer {
    index: u32,
    type_: u32,
    bytesused: u32,
    flags: u32,
    field: u32,
    timestamp: V4l2Timeval,
    timecode: [u8; 16],
    sequence: u32,
    memory: u32,
    offset: u32, // m.offset in union
    length: u32,
    reserved2: u32,
    request_fd: i32,
}

// =============================================================================
// Memory-Mapped Buffer
// =============================================================================

struct MmapBuffer {
    ptr: *mut u8,
    length: usize,
}

impl Drop for MmapBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                libc::munmap(self.ptr as *mut libc::c_void, self.length);
            }
        }
    }
}

// =============================================================================
// V4L2 Backend
// =============================================================================

/// V4L2 camera capture backend for Linux.
///
/// Uses memory-mapped I/O for zero-copy frame capture from USB cameras
/// and CSI cameras (Raspberry Pi).
///
/// # Example
/// ```ignore
/// use axonml_vision::camera::{V4L2Backend, CaptureBackend, CaptureConfig, PixelFormat};
///
/// let mut cam = V4L2Backend::new("/dev/video0");
/// cam.open(&CaptureConfig {
///     width: 640,
///     height: 480,
///     format: PixelFormat::Yuyv,
///     fps: 30,
/// }).unwrap();
///
/// let frame = cam.grab_frame().unwrap();
/// cam.close();
/// ```
pub struct V4L2Backend {
    device_path: String,
    file: Option<fs::File>,
    buffers: Vec<MmapBuffer>,
    width: u32,
    height: u32,
    format: PixelFormat,
    streaming: bool,
}

impl V4L2Backend {
    /// Create a V4L2 backend for the specified device.
    ///
    /// Common paths: `/dev/video0`, `/dev/video1`.
    pub fn new(device_path: &str) -> Self {
        Self {
            device_path: device_path.to_string(),
            file: None,
            buffers: Vec::new(),
            width: 0,
            height: 0,
            format: PixelFormat::Yuyv,
            streaming: false,
        }
    }

    /// Open the default camera device (`/dev/video0`).
    pub fn default_device() -> Self {
        Self::new("/dev/video0")
    }

    fn fd(&self) -> Result<i32, CaptureError> {
        self.file
            .as_ref()
            .map(|f| f.as_raw_fd())
            .ok_or(CaptureError::NotOpen)
    }

    fn ioctl(&self, request: libc::c_ulong, arg: *mut libc::c_void) -> Result<(), CaptureError> {
        let fd = self.fd()?;
        let ret = unsafe { libc::ioctl(fd, request, arg) };
        if ret < 0 {
            Err(CaptureError::CaptureError(format!(
                "ioctl 0x{:X} failed: {}",
                request,
                std::io::Error::last_os_error()
            )))
        } else {
            Ok(())
        }
    }

    fn set_format(&mut self, config: &CaptureConfig) -> Result<(), CaptureError> {
        let pixfmt = match config.format {
            PixelFormat::Yuyv => V4L2_PIX_FMT_YUYV,
            PixelFormat::Mjpeg => V4L2_PIX_FMT_MJPEG,
            _ => V4L2_PIX_FMT_YUYV,
        };

        let mut fmt: V4l2Format = unsafe { std::mem::zeroed() };
        fmt.type_ = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        fmt.fmt.width = config.width;
        fmt.fmt.height = config.height;
        fmt.fmt.pixelformat = pixfmt;

        self.ioctl(VIDIOC_S_FMT, &mut fmt as *mut _ as *mut libc::c_void)?;

        self.width = fmt.fmt.width;
        self.height = fmt.fmt.height;
        self.format = config.format;

        Ok(())
    }

    fn request_buffers(&mut self) -> Result<(), CaptureError> {
        let mut req: V4l2RequestBuffers = unsafe { std::mem::zeroed() };
        req.count = NUM_BUFFERS;
        req.type_ = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        req.memory = V4L2_MEMORY_MMAP;

        self.ioctl(VIDIOC_REQBUFS, &mut req as *mut _ as *mut libc::c_void)?;

        let fd = self.fd()?;

        for i in 0..req.count {
            let mut buf: V4l2Buffer = unsafe { std::mem::zeroed() };
            buf.index = i;
            buf.type_ = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf.memory = V4L2_MEMORY_MMAP;

            self.ioctl(VIDIOC_QUERYBUF, &mut buf as *mut _ as *mut libc::c_void)?;

            let ptr = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    buf.length as usize,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_SHARED,
                    fd,
                    buf.offset as libc::off_t,
                )
            };

            if ptr == libc::MAP_FAILED {
                return Err(CaptureError::CaptureError("mmap failed".into()));
            }

            self.buffers.push(MmapBuffer {
                ptr: ptr as *mut u8,
                length: buf.length as usize,
            });

            // Queue the buffer
            self.ioctl(VIDIOC_QBUF, &mut buf as *mut _ as *mut libc::c_void)?;
        }

        Ok(())
    }

    fn start_streaming(&mut self) -> Result<(), CaptureError> {
        let mut type_ = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        self.ioctl(VIDIOC_STREAMON, &mut type_ as *mut _ as *mut libc::c_void)?;
        self.streaming = true;
        Ok(())
    }

    fn stop_streaming(&mut self) {
        if self.streaming {
            let mut type_ = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            let _ = self.ioctl(VIDIOC_STREAMOFF, &mut type_ as *mut _ as *mut libc::c_void);
            self.streaming = false;
        }
    }
}

impl CaptureBackend for V4L2Backend {
    fn open(&mut self, config: &CaptureConfig) -> Result<(), CaptureError> {
        use std::fs::OpenOptions;

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&self.device_path)
            .map_err(|e| CaptureError::DeviceNotFound(format!("{}: {}", self.device_path, e)))?;

        self.file = Some(file);

        // Query capabilities
        let mut cap: V4l2Capability = unsafe { std::mem::zeroed() };
        self.ioctl(VIDIOC_QUERYCAP, &mut cap as *mut _ as *mut libc::c_void)?;

        self.set_format(config)?;
        self.request_buffers()?;
        self.start_streaming()?;

        Ok(())
    }

    fn grab_frame(&mut self) -> Result<FrameBuffer, CaptureError> {
        if !self.streaming {
            return Err(CaptureError::NotOpen);
        }

        // Dequeue a buffer
        let mut buf: V4l2Buffer = unsafe { std::mem::zeroed() };
        buf.type_ = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;

        self.ioctl(VIDIOC_DQBUF, &mut buf as *mut _ as *mut libc::c_void)?;

        let idx = buf.index as usize;
        let mmap = &self.buffers[idx];

        // Copy frame data
        let data = unsafe {
            std::slice::from_raw_parts(mmap.ptr, buf.bytesused as usize)
        };
        let frame_data = data.to_vec();

        let timestamp_us = (buf.timestamp.tv_sec as u64) * 1_000_000
            + (buf.timestamp.tv_usec as u64);

        // Re-queue the buffer
        self.ioctl(VIDIOC_QBUF, &mut buf as *mut _ as *mut libc::c_void)?;

        Ok(FrameBuffer {
            data: frame_data,
            width: self.width,
            height: self.height,
            format: self.format,
            timestamp_us,
        })
    }

    fn is_open(&self) -> bool {
        self.streaming
    }

    fn close(&mut self) {
        self.stop_streaming();
        self.buffers.clear();
        self.file = None;
    }

    fn resolution(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}

impl Drop for V4L2Backend {
    fn drop(&mut self) {
        self.close();
    }
}

// Note: V4L2 tests require actual camera hardware and are not included
// in the unit test suite. Integration testing is done on target devices.

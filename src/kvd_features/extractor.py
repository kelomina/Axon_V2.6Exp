"""KVD 恶意软件特征提取器核心模块。

本模块提供完整的恶意软件特征提取功能，包括：
- 字节序列提取（固定长度）
- PE 结构特征提取（1500维）
- 统计特征提取（~100维）
- 轻量级哈希特征（256维）

特征维度：
- 字节序列：max_file_size (默认 65536)
- PE结构特征：1500
- 统计特征：~100
- 轻量级哈希：256
"""

import os
import hashlib
import struct
from typing import Tuple, Optional
from dataclasses import dataclass

import numpy as np

try:
    import pefile
    PEFILE_AVAILABLE = True
except ImportError:
    PEFILE_AVAILABLE = False


@dataclass
class ExtractionConfig:
    """特征提取配置"""
    max_file_size: int = 65536  # 64KB
    byte_histogram_bins: int = 256
    stat_chunk_count: int = 10
    entropy_block_size: int = 4096
    entropy_sample_size: int = 4096
    size_norm_max: int = 100 * 1024 * 1024  # 100MB
    timestamp_year_base: int = 1970
    timestamp_year_max: int = 2099
    large_trailing_data_size: int = 1024 * 1024  # 1MB
    entropy_high_threshold: float = 0.8
    section_entropy_min_size: int = 256
    overlay_entropy_min_size: int = 256

    # PE特征维度
    pe_feature_dim: int = 1500
    
    # 轻量级哈希维度
    lightweight_feature_dim: int = 256


# 特征名称列表（对应 PE_FEATURE_VECTOR_DIM）
FEATURE_NAMES = [
    'size', 'log_size', 'entropy', 'section_entropy_max', 'section_entropy_min',
    'section_entropy_avg', 'section_entropy_std', 'packed_sections_ratio',
    'sections_count', 'section_total_size', 'section_total_vsize',
    'avg_section_size', 'avg_section_vsize', 'min_section_size', 'max_section_size',
    'section_size_std', 'section_size_cv', 'section_names_count',
    'section_name_avg_length', 'section_name_max_length', 'section_name_min_length',
    'long_sections_count', 'long_sections_ratio', 'short_sections_count', 'short_sections_ratio',
    'executable_sections_ratio', 'writable_sections_ratio', 'readable_sections_ratio',
    'rwx_sections_ratio', 'rwx_sections_count', 'executable_code_density',
    'executable_writable_sections', 'non_standard_executable_sections_count',
    'non_standard_executable_sections_ratio', 'imports_count', 'unique_imports',
    'unique_dlls', 'import_ordinal_only_count', 'import_ordinal_only_ratio',
    'avg_imports_per_dll', 'imported_system_dlls_count', 'imported_system_dlls_ratio',
    'dll_name_avg_length', 'dll_name_max_length', 'dll_name_min_length',
    'dll_imports_entropy', 'api_imports_entropy', 'imports_per_section',
    'syscall_api_ratio', 'exports_count', 'exports_density', 'export_name_avg_length',
    'export_name_max_length', 'export_name_min_length', 'exports_name_ratio',
    'has_resources', 'resources_count', 'resource_types_count', 'pe_header_size',
    'header_size_ratio', 'subsystem', 'dll_characteristics', 'checksum',
    'checksum_zero_flag', 'has_aslr', 'has_nx_compat', 'has_guard_cf', 'has_seh',
    'has_debug_info', 'has_relocs', 'has_tls', 'has_exceptions', 'has_signature',
    'entry_point_ratio', 'entry_in_nonstandard_section_flag', 'trailing_data_size',
    'trailing_data_ratio', 'has_large_trailing_data', 'overlay_entropy',
    'overlay_high_entropy_flag', 'tls_callbacks_count', 'reloc_blocks_count',
    'reloc_entries_count', 'alignment_mismatch_count', 'alignment_mismatch_ratio',
    'api_network_ratio', 'api_process_ratio', 'api_filesystem_ratio',
    'api_registry_ratio', 'packer_keyword_hits_count', 'packer_keyword_hits_ratio',
]


def extract_byte_sequence(
    file_path: str, 
    max_file_size: int = 65536
) -> Tuple[Optional[np.ndarray], int]:
    """从文件中提取固定长度的字节序列。
    
    Args:
        file_path: 文件路径
        max_file_size: 最大读取字节数
        
    Returns:
        Tuple of (字节序列 numpy 数组, 原始文件长度)
        如果失败则返回 (None, 0)
    """
    try:
        with open(file_path, 'rb') as f:
            raw_bytes = f.read()
        
        orig_len = len(raw_bytes)
        
        if orig_len > max_file_size:
            padded_sequence = np.frombuffer(raw_bytes[:max_file_size], dtype=np.uint8).copy()
            return padded_sequence, orig_len
        else:
            padded_sequence = np.zeros(max_file_size, dtype=np.uint8)
            padded_sequence[:orig_len] = np.frombuffer(raw_bytes, dtype=np.uint8)
            return padded_sequence, orig_len
            
    except Exception as e:
        print(f"[Error] Failed to extract byte sequence: {e}")
        return None, 0


def calculate_byte_entropy(
    byte_sequence: np.ndarray, 
    block_size: int = 4096
) -> float:
    """计算字节序列的熵值。
    
    Args:
        byte_sequence: 字节序列
        block_size: 块大小
        
    Returns:
        熵值（0-1之间）
    """
    if byte_sequence is None or len(byte_sequence) == 0:
        return 0.0
    
    hist = np.bincount(byte_sequence, minlength=256)
    prob = hist / len(byte_sequence)
    prob = prob[prob > 0]
    
    if len(prob) == 0:
        return 0.0
    
    entropy = -np.sum(prob * np.log2(prob)) / 8.0
    return float(entropy)


def extract_statistical_features(
    byte_sequence: np.ndarray,
    pe_features: np.ndarray,
    orig_length: Optional[int] = None
) -> np.ndarray:
    """提取字节序列的统计特征。
    
    Args:
        byte_sequence: 字节序列 [max_file_size]
        pe_features: PE结构特征 [1500]
        orig_length: 原始文件长度
        
    Returns:
        统计特征向量
    """
    if orig_length is not None and orig_length >= 0:
        byte_array = np.asarray(byte_sequence[:orig_length], dtype=np.uint8)
    else:
        byte_array = np.asarray(byte_sequence, dtype=np.uint8)
    
    length = len(byte_array)
    features = []
    
    # 基本统计
    counts = np.bincount(byte_array, minlength=256) if length > 0 else np.zeros(256, dtype=np.int64)
    
    if length > 0:
        value_axis = np.arange(256, dtype=np.float64)
        weighted_sum = float(np.dot(counts, value_axis))
        mean_val = weighted_sum / float(length)
        weighted_sq_sum = float(np.dot(counts, value_axis * value_axis))
        variance = max(0.0, weighted_sq_sum / float(length) - mean_val * mean_val)
        std_val = float(np.sqrt(variance))
        
        nonzero_indices = np.flatnonzero(counts)
        min_val = float(nonzero_indices[0]) if nonzero_indices.size > 0 else 0.0
        max_val = float(nonzero_indices[-1]) if nonzero_indices.size > 0 else 0.0
        
        cdf = np.cumsum(counts)
        median_val = float(np.searchsorted(cdf, int(np.ceil(0.50 * length))))
        q25 = float(np.searchsorted(cdf, int(np.ceil(0.25 * length))))
        q75 = float(np.searchsorted(cdf, int(np.ceil(0.75 * length))))
    else:
        mean_val = std_val = min_val = max_val = median_val = q25 = q75 = 0.0
    
    features.extend([mean_val, std_val, min_val, max_val, median_val, q25, q75])
    
    # 字节计数
    features.extend([
        int(counts[0]),
        int(counts[255]),
        int(counts[0x90]),
        int(np.sum(counts[32:127])),  # 可打印字符
    ])
    
    # 熵值
    p = counts.astype(np.float64) / float(length) if length > 0 else np.zeros_like(counts, dtype=np.float64)
    p = p[p > 0]
    entropy = float((-np.sum(p * np.log2(p)) / 8.0) if p.size > 0 else 0.0)
    features.append(entropy)
    
    # 分段统计（三等分）
    if length >= 3:
        one_third = length // 3
        segments = [
            byte_array[:one_third].copy(),
            byte_array[one_third:2 * one_third].copy(),
            byte_array[2 * one_third:].copy(),
        ]
    else:
        segments = [byte_array.copy(), byte_array.copy(), byte_array.copy()]
    
    for seg in segments:
        if len(seg) == 0:
            seg_mean = seg_std = seg_entropy = 0.0
        else:
            seg_mean = float(np.mean(seg))
            seg_std = float(np.std(seg))
            seg_counts = np.bincount(seg, minlength=256)
            seg_p = seg_counts.astype(np.float64) / float(len(seg))
            seg_p = seg_p[seg_p > 0]
            seg_entropy = float((-np.sum(seg_p * np.log2(seg_p)) / 8.0) if seg_p.size > 0 else 0.0)
        features.extend([seg_mean, seg_std, seg_entropy])
    
    # 分块统计
    chunk_size = max(1, length // 10)
    chunk_means = []
    chunk_stds = []
    
    for i in range(10):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size if i < 9 else length
        chunk = byte_array[start_idx:end_idx]
        
        if len(chunk) > 0:
            chunk_means.append(float(np.mean(chunk)))
            chunk_stds.append(float(np.std(chunk)))
        else:
            chunk_means.append(0.0)
            chunk_stds.append(0.0)
    
    features.extend(chunk_means)
    features.extend(chunk_stds)
    
    # 块间差异统计
    chunk_means = np.array(chunk_means, dtype=np.float32)
    chunk_stds = np.array(chunk_stds, dtype=np.float32)
    
    if len(chunk_means) > 1:
        mean_diffs = np.diff(chunk_means)
        std_diffs = np.diff(chunk_stds)
        
        features.extend([
            float(np.mean(np.abs(mean_diffs))),
            float(np.std(mean_diffs)),
            float(np.max(mean_diffs)),
            float(np.min(mean_diffs)),
            float(np.mean(np.abs(std_diffs))),
            float(np.std(std_diffs)),
            float(np.max(std_diffs)),
            float(np.min(std_diffs)),
        ])
    else:
        features.extend([0.0] * 8)
    
    # 添加 PE 特征
    features.extend(pe_features.tolist())
    
    return np.array(features, dtype=np.float32)


class PEFeatureExtractor:
    """PE结构特征提取器。
    
    提取 1500 维的 PE 结构特征，包括：
    - 文件统计特征
    - PE Header 特征
    - Section 特征
    - 导入表特征
    - 导出表特征
    - 安全标志特征
    - 尾部数据特征
    - 资源特征
    """
    
    def __init__(self, config: Optional[ExtractionConfig] = None):
        self.config = config or ExtractionConfig()
        
        # 常见 section 名称
        self.common_sections = {
            '.text', '.data', '.rdata', '.bss', '.rsrc', '.reloc',
            '.textbss', '.code', '.idata', '.edata', '.tls', '.stub'
        }
        
        # 系统 DLL 列表
        self.system_dlls = {
            'kernel32.dll', 'user32.dll', 'advapi32.dll', 'shell32.dll',
            'ole32.dll', 'oleaut32.dll', 'msvcrt.dll', 'ntdll.dll',
            'ws2_32.dll', 'wininet.dll', 'urlmon.dll', 'crypt32.dll',
            'secur32.dll', 'netapi32.dll', 'dnsapi.dll', 'iphlpapi.dll',
            'gdi32.dll', 'comdlg32.dll', 'comctl32.dll', 'shlwapi.dll',
            'version.dll', 'setupapi.dll', 'imm32.dll', 'midimap.dll',
            'msacm32.dll', 'ddraw.dll', 'dinput.dll', 'dsound.dll'
        }
        
        # 打包器特征关键字
        self.packer_keywords = {
            'upx', 'aspack', 'petite', 'pecompact', 'themida', 'vmprotect',
            'themida', 'enigma', 'obsidium', 'armadillo', 'safengine',
            'orion', 'execryptor', ' PELock', 'npack', 'nspack', 'wwpack',
            'diminuto', 'petite', 'upack', 'kkrunchy', 'joexe', ' fsg',
            'stunnix', 'winlicense', 'themida'
        }
        
        # API 类别关键字
        self.api_categories = {
            'network': ['Internet', 'Http', 'Socket', 'Connect', 'Recv', 'Send', 
                       'URL', 'Download', 'Upload', 'Proxy', 'WSA', 'FTP', 'SMTP'],
            'process': ['CreateProcess', 'OpenProcess', 'VirtualAlloc', 'VirtualProtect',
                       'WriteProcessMemory', 'ReadProcessMemory', 'CreateRemoteThread',
                       'ShellExecute', 'WinExec', 'LoadLibrary', 'GetProcAddress'],
            'filesystem': ['CreateFile', 'ReadFile', 'WriteFile', 'DeleteFile',
                         'MoveFile', 'CopyFile', 'GetFileSize', 'SetFilePointer',
                         'FindFirstFile', 'FindNextFile', 'GetTempPath'],
            'registry': ['RegOpenKey', 'RegSetValue', 'RegCreateKey', 'RegDeleteKey',
                        'RegQueryValue', 'RegCloseKey', 'SaveKey', 'RestoreKey']
        }
    
    def _safe_dll_name(self, dll_name: bytes) -> str:
        """安全处理 DLL 名称"""
        try:
            return dll_name.decode('utf-8', errors='ignore').lower().strip()
        except:
            return ""
    
    def _safe_api_name(self, api_name: bytes) -> str:
        """安全处理 API 名称"""
        try:
            return api_name.decode('utf-8', errors='ignore').lower().strip()
        except:
            return ""
    
    def extract(self, file_path: str) -> Optional[np.ndarray]:
        """提取 PE 文件特征。
        
        Args:
            file_path: PE 文件路径
            
        Returns:
            1500 维特征向量，失败返回 None
        """
        if not PEFILE_AVAILABLE:
            return self._extract_fallback(file_path)
        
        try:
            pe = pefile.PE(file_path)
            features = np.zeros(self.config.pe_feature_dim, dtype=np.float32)
            
            idx = 0
            
            # 1. 文件级统计特征
            file_size = os.path.getsize(file_path)
            features[idx] = float(file_size); idx += 1
            features[idx] = np.log1p(float(file_size)); idx += 1
            
            # 2. PE Header 特征
            features[idx] = pe.FILE_HEADER.SizeOfOptionalHeader; idx += 1
            header_size = pe.FILE_HEADER.SizeOfOptionalHeader + 24  # DOS + PE
            features[idx] = header_size / max(file_size, 1); idx += 1
            features[idx] = float(pe.OPTIONAL_HEADER.Subsystem); idx += 1
            features[idx] = float(pe.OPTIONAL_HEADER.DllCharacteristics); idx += 1
            features[idx] = float(pe.OPTIONAL_HEADER.CheckSum); idx += 1
            features[idx] = 1.0 if pe.OPTIONAL_HEADER.CheckSum == 0 else 0.0; idx += 1
            
            # 3. 安全标志
            dll_chars = pe.OPTIONAL_HEADER.DllCharacteristics
            features[idx] = 1.0 if (dll_chars & 0x0040) else 0.0; idx += 1  # ASLR
            features[idx] = 1.0 if (dll_chars & 0x0080) else 0.0; idx += 1  # NX
            features[idx] = 1.0 if (dll_chars & 0x4000) else 0.0; idx += 1  # CFG
            features[idx] = 1.0 if (pe.FILE_HEADER.Characteristics & 0x0004) else 0.0; idx += 1  # SEH
            features[idx] = 1.0 if hasattr(pe, 'DIRECTORY_ENTRY_DEBUG') else 0.0; idx += 1
            features[idx] = 1.0 if hasattr(pe, 'DIRECTORY_ENTRY_BASERELOC') else 0.0; idx += 1
            features[idx] = 1.0 if hasattr(pe, 'DIRECTORY_ENTRY_TLS') else 0.0; idx += 1
            features[idx] = 1.0 if hasattr(pe, 'DIRECTORY_ENTRY_EXCEPTION') else 0.0; idx += 1
            features[idx] = 1.0 if hasattr(pe, 'DIRECTORY_ENTRY_SECURITY') else 0.0; idx += 1
            
            # 4. Section 特征
            num_sections = pe.FILE_HEADER.NumberOfSections
            features[idx] = float(num_sections); idx += 1
            
            section_sizes = []
            section_entropies = []
            section_vsize = []
            
            for section in pe.sections:
                section_name = section.Name.decode('utf-8', errors='ignore').strip('\x00')
                raw_size = section.SizeOfRawData
                virt_size = section.Misc_VirtualSize
                
                section_sizes.append(raw_size)
                section_vsize.append(virt_size)
                
                # 计算 section 熵
                if raw_size > 0 and raw_size < 10 * 1024 * 1024:  # < 10MB
                    section_data = section.get_data()
                    if len(section_data) > 0:
                        entropy = calculate_byte_entropy(
                            np.frombuffer(section_data[:self.config.section_entropy_min_size], dtype=np.uint8)
                        )
                        section_entropies.append(entropy)
                
                # Section 属性
                chars = section.Characteristics
                is_exec = bool(chars & 0x20000000)
                is_write = bool(chars & 0x80000000)
                is_read = bool(chars & 0x40000000)
                
                features[idx] = 1.0 if is_exec else 0.0; idx += 1
                features[idx] = 1.0 if is_write else 0.0; idx += 1
                features[idx] = 1.0 if is_read else 0.0; idx += 1
            
            # Section 统计
            if section_entropies:
                features[idx] = max(section_entropies); idx += 1
                features[idx] = min(section_entropies); idx += 1
                features[idx] = np.mean(section_entropies); idx += 1
                features[idx] = np.std(section_entropies); idx += 1
                high_entropy_ratio = sum(1 for e in section_entropies if e > self.config.entropy_high_threshold) / len(section_entropies)
                features[idx] = high_entropy_ratio; idx += 1
            else:
                features[idx:idx+5] = 0.0; idx += 5
            
            if section_sizes:
                total_raw = sum(section_sizes)
                total_vsize = sum(section_vsize)
                avg_raw = np.mean(section_sizes)
                avg_vsize = np.mean(section_vsize)
                
                features[idx] = float(total_raw); idx += 1
                features[idx] = float(total_vsize); idx += 1
                features[idx] = float(avg_raw); idx += 1
                features[idx] = float(avg_vsize); idx += 1
                features[idx] = float(min(section_sizes)); idx += 1
                features[idx] = float(max(section_sizes)); idx += 1
                features[idx] = float(np.std(section_sizes)); idx += 1
                features[idx] = float(np.std(section_sizes) / max(avg_raw, 1)); idx += 1
            else:
                features[idx:idx+8] = 0.0; idx += 8
            
            # Section 名称统计
            section_names = [s.Name.decode('utf-8', errors='ignore').strip('\x00') for s in pe.sections]
            valid_names = [n for n in section_names if n]
            features[idx] = len(valid_names); idx += 1
            if valid_names:
                name_lens = [len(n) for n in valid_names]
                features[idx] = np.mean(name_lens); idx += 1
                features[idx] = max(name_lens); idx += 1
                features[idx] = min(name_lens); idx += 1
            else:
                features[idx:idx+3] = 0.0; idx += 3
            
            # 大小异常 section
            if section_sizes and avg_raw > 0:
                long_count = sum(1 for s in section_sizes if s > 2 * avg_raw)
                short_count = sum(1 for s in section_sizes if s < avg_raw / 2)
                features[idx] = float(long_count); idx += 1
                features[idx] = float(long_count / len(section_sizes)); idx += 1
                features[idx] = float(short_count); idx += 1
                features[idx] = float(short_count / len(section_sizes)); idx += 1
            else:
                features[idx:idx+4] = 0.0; idx += 4
            
            # 填充剩余维度
            while idx < self.config.pe_feature_dim:
                features[idx] = 0.0
                idx += 1
            
            return features
            
        except Exception as e:
            print(f"[Error] PE extraction failed: {e}")
            return self._extract_fallback(file_path)
    
    def _extract_fallback(self, file_path: str) -> Optional[np.ndarray]:
        """PE 提取失败时的降级处理"""
        try:
            with open(file_path, 'rb') as f:
                data = f.read(1024 * 1024)  # 只读前 1MB
            
            features = np.zeros(self.config.pe_feature_dim, dtype=np.float32)
            
            # 基本统计
            file_size = os.path.getsize(file_path)
            features[0] = float(file_size)
            features[1] = np.log1p(float(file_size))
            
            # 简单熵值
            byte_arr = np.frombuffer(data, dtype=np.uint8)
            features[2] = calculate_byte_entropy(byte_arr)
            
            return features
        except:
            return None


def extract_lightweight_features(
    file_path: str,
    feature_dim: int = 256
) -> np.ndarray:
    """提取轻量级哈希特征。
    
    基于 DLL 名称、API 函数名和 Section 名称的哈希映射。
    
    Args:
        file_path: 文件路径
        feature_dim: 特征维度（默认 256）
        
    Returns:
        256 维二值特征向量
    """
    features = np.zeros(feature_dim, dtype=np.float32)
    
    try:
        # 读取文件前 64KB
        with open(file_path, 'rb') as f:
            data = f.read(65536)
        
        data_lower = data.lower()
        
        # DLL 名称哈希 (0-127)
        dll_patterns = [
            b'kernel32.dll', b'user32.dll', b'ntdll.dll', b'advapi32.dll',
            b'ws2_32.dll', b'wininet.dll', b'ole32.dll', b'shell32.dll',
            b'msvcrt.dll', b'msvcrtd.dll', b'vcruntime.dll', b'ucrtbase.dll',
        ]
        
        for pattern in dll_patterns:
            if pattern in data_lower:
                h = hashlib.sha256(pattern).digest()[0]
                features[h % 128] = 1
        
        # API 函数名哈希 (128-255)
        api_patterns = [
            b'VirtualAlloc', b'VirtualProtect', b'CreateRemoteThread',
            b'WriteProcessMemory', b'ReadProcessMemory', b'WinExec',
            b'ShellExecute', b'LoadLibrary', b'GetProcAddress',
            b'CreateProcess', b'InternetOpen', b'InternetReadFile',
            b'URLDownloadToFile', b'CreateFile', b'RegOpenKey',
        ]
        
        for pattern in api_patterns:
            if pattern in data_lower:
                h = hashlib.sha256(pattern).digest()[0]
                features[128 + (h % 128)] = 1
        
        # Section 名称哈希 (224-255)
        section_patterns = [
            b'.text', b'.data', b'.rdata', b'.rsrc', b'.reloc',
            b'.code', b'.idata', b'.edata', b'.tls', b'.bss',
        ]
        
        for pattern in section_patterns:
            if pattern in data_lower:
                h = hashlib.sha256(pattern).digest()[0]
                features[(h % 32) + 224] = 1
        
        # L2 归一化
        norm = np.linalg.norm(features)
        if norm > 0:
            features /= norm
            
    except Exception as e:
        print(f"[Error] Lightweight feature extraction failed: {e}")
    
    return features


def extract_pe_features(file_path: str) -> Optional[np.ndarray]:
    """提取 PE 结构特征。
    
    Args:
        file_path: 文件路径
        
    Returns:
        1500 维特征向量
    """
    extractor = PEFeatureExtractor()
    return extractor.extract(file_path)


def extract_all_features(
    file_path: str,
    config: Optional[ExtractionConfig] = None
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], int]:
    """提取所有特征。
    
    Args:
        file_path: 文件路径
        config: 提取配置
        
    Returns:
        Tuple of (字节序列, PE特征, 统计特征, 原始长度)
    """
    config = config or ExtractionConfig()
    
    # 字节序列
    byte_seq, orig_len = extract_byte_sequence(file_path, config.max_file_size)
    if byte_seq is None:
        return None, None, None, 0
    
    # PE 特征
    pe_features = extract_pe_features(file_path)
    if pe_features is None:
        pe_features = np.zeros(config.pe_feature_dim, dtype=np.float32)
    
    # 统计特征
    stat_features = extract_statistical_features(byte_seq, pe_features, orig_len)
    
    return byte_seq, pe_features, stat_features, orig_len


# 兼容性别名
FeatureExtractor = PEFeatureExtractor

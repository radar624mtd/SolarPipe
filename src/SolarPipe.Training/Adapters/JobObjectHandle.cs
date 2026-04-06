using System.Diagnostics;
using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;

namespace SolarPipe.Training.Adapters;

// Thin Windows Job Object wrapper that ensures child process death when the .NET host exits.
// RULE-060: Orphan-prevention mechanism for the Python sidecar on Windows.
//
// On non-Windows platforms this type is never instantiated (caller guards with OperatingSystem.IsWindows()).
[System.Runtime.Versioning.SupportedOSPlatform("windows")]
internal sealed class JobObjectHandle : IDisposable
{
    private readonly SafeFileHandle _handle;

    private JobObjectHandle(SafeFileHandle handle) => _handle = handle;

    public static JobObjectHandle CreateAndAssign(Process process)
    {
        var handle = NativeMethods.CreateJobObject(IntPtr.Zero, null);
        if (handle.IsInvalid)
            throw new InvalidOperationException(
                $"CreateJobObject failed: HRESULT={Marshal.GetLastWin32Error()} " +
                $"(stage=SidecarLifecycleService, pid={process.Id}).");

        var info = new NativeMethods.JOBOBJECT_EXTENDED_LIMIT_INFORMATION
        {
            BasicLimitInformation = new NativeMethods.JOBOBJECT_BASIC_LIMIT_INFORMATION
            {
                LimitFlags = NativeMethods.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
            },
        };

        int infoSize = Marshal.SizeOf<NativeMethods.JOBOBJECT_EXTENDED_LIMIT_INFORMATION>();
        IntPtr infoPtr = Marshal.AllocHGlobal(infoSize);
        try
        {
            Marshal.StructureToPtr(info, infoPtr, false);
            if (!NativeMethods.SetInformationJobObject(
                    handle,
                    NativeMethods.JobObjectInfoType.ExtendedLimitInformation,
                    infoPtr, (uint)infoSize))
            {
                throw new InvalidOperationException(
                    $"SetInformationJobObject failed: HRESULT={Marshal.GetLastWin32Error()} " +
                    $"(stage=SidecarLifecycleService, pid={process.Id}).");
            }
        }
        finally
        {
            Marshal.FreeHGlobal(infoPtr);
        }

        if (!NativeMethods.AssignProcessToJobObject(handle, process.Handle))
            throw new InvalidOperationException(
                $"AssignProcessToJobObject failed: HRESULT={Marshal.GetLastWin32Error()} " +
                $"(stage=SidecarLifecycleService, pid={process.Id}).");

        return new JobObjectHandle(handle);
    }

    public void Dispose() => _handle.Dispose();

    private static class NativeMethods
    {
        [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Unicode)]
        public static extern SafeFileHandle CreateJobObject(IntPtr lpJobAttributes, string? lpName);

        [DllImport("kernel32.dll", SetLastError = true)]
        public static extern bool SetInformationJobObject(
            SafeFileHandle hJob,
            JobObjectInfoType infoType,
            IntPtr lpJobObjectInfo,
            uint cbJobObjectInfoLength);

        [DllImport("kernel32.dll", SetLastError = true)]
        public static extern bool AssignProcessToJobObject(SafeFileHandle hJob, IntPtr hProcess);

        public const uint JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000;

        public enum JobObjectInfoType
        {
            ExtendedLimitInformation = 9,
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct JOBOBJECT_BASIC_LIMIT_INFORMATION
        {
            public long PerProcessUserTimeLimit;
            public long PerJobUserTimeLimit;
            public uint LimitFlags;
            public UIntPtr MinimumWorkingSetSize;
            public UIntPtr MaximumWorkingSetSize;
            public uint ActiveProcessLimit;
            public UIntPtr Affinity;
            public uint PriorityClass;
            public uint SchedulingClass;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct IO_COUNTERS
        {
            public ulong ReadOperationCount;
            public ulong WriteOperationCount;
            public ulong OtherOperationCount;
            public ulong ReadTransferCount;
            public ulong WriteTransferCount;
            public ulong OtherTransferCount;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct JOBOBJECT_EXTENDED_LIMIT_INFORMATION
        {
            public JOBOBJECT_BASIC_LIMIT_INFORMATION BasicLimitInformation;
            public IO_COUNTERS IoInfo;
            public UIntPtr ProcessMemoryLimit;
            public UIntPtr JobMemoryLimit;
            public UIntPtr PeakProcessMemoryUsed;
            public UIntPtr PeakJobMemoryUsed;
        }
    }
}

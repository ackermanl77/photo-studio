// PhotoStudio.FilterHost: runs third-party Photoshop filter plug-ins (.8bf) for Photo Studio, in a
// process of its own so a misbehaving plug-in can't take the editor down (app/filter_plugins.py
// starts it). Built on PSFilterHost (Ms-PL, https://github.com/0xC0000054/PSFilterHost), which
// implements the Photoshop plug-in API.
//
//   PhotoStudio.FilterHost list <folder> [--recursive]
//       JSON array on stdout: [{"file","entry","category","title","about"}...]
//   PhotoStudio.FilterHost run --plugin <file.8bf> --entry <entry point> --input <in.png> --output <out.png>
//                              [--params-in <file>] [--params-out <file>] [--ui]
//       --ui shows the filter's own dialog; without it the filter re-runs with --params-in's settings.
//       --params-out saves the settings it ran with, to replay later without the dialog.
//
// Exit codes: 0 applied, 3 cancelled in the filter's dialog, 2 error (message on stderr).
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using PSFilterHostDll;

namespace PhotoStudio.FilterHost
{
    internal static class Program
    {
        private const int Applied = 0;
        private const int Failed = 2;
        private const int Cancelled = 3;

        [STAThread]
        private static int Main(string[] args)
        {
            try
            {
                if (args.Length >= 2 && args[0] == "list")
                {
                    return List(args[1], args.Contains("--recursive"));
                }
                if (args.Length >= 1 && args[0] == "run")
                {
                    return Run(ParseOptions(args.Skip(1).ToArray()));
                }
                Console.Error.WriteLine("Usage: PhotoStudio.FilterHost list <folder> [--recursive] | run --plugin ... (see Program.cs)");
                return Failed;
            }
            catch (Exception error)
            {
                Console.Error.WriteLine(error.GetType().Name + ": " + error.Message);
                return Failed;
            }
        }

        private static Dictionary<string, string> ParseOptions(string[] args)
        {
            var options = new Dictionary<string, string>(StringComparer.Ordinal);
            for (int i = 0; i < args.Length; i++)
            {
                if (args[i] == "--ui")
                {
                    options["ui"] = "1";
                }
                else if (args[i].StartsWith("--", StringComparison.Ordinal) && i + 1 < args.Length)
                {
                    options[args[i].Substring(2)] = args[++i];
                }
            }
            return options;
        }

        private static int List(string folder, bool recursive)
        {
            var output = new StringBuilder("[");
            bool first = true;
            foreach (PluginData plugin in PSFilterHost.EnumerateFilters(folder, recursive))
            {
                output.Append(first ? "" : ",");
                first = false;
                output.Append("{\"file\":").Append(Json(plugin.FileName))
                      .Append(",\"entry\":").Append(Json(plugin.EntryPoint))
                      .Append(",\"category\":").Append(Json(plugin.Category))
                      .Append(",\"title\":").Append(Json(plugin.Title))
                      .Append(",\"about\":").Append(plugin.HasAboutBox ? "true" : "false").Append('}');
            }
            Console.OutputEncoding = Encoding.UTF8;
            Console.WriteLine(output.Append(']').ToString());
            return Applied;
        }

        private static int Run(Dictionary<string, string> options)
        {
            string pluginFile = Required(options, "plugin");
            string entry = Required(options, "entry");
            PluginData plugin = PSFilterHost.EnumerateFilters(Path.GetDirectoryName(pluginFile), false)
                .FirstOrDefault(p => string.Equals(p.FileName, pluginFile, StringComparison.OrdinalIgnoreCase) && p.EntryPoint == entry);
            if (plugin == null)
            {
                throw new FileNotFoundException("No filter " + entry + " in " + pluginFile + ".");
            }

            BitmapSource source = ReadPng(Required(options, "input"));
            using (var host = new PSFilterHost(source, IntPtr.Zero))
            {
                string paramsIn;
                if (options.TryGetValue("params-in", out paramsIn))
                {
                    host.FilterParameters = ReadParameters(paramsIn);
                }
                bool showUI = options.ContainsKey("ui");
                if (!host.RunFilter(plugin, showUI))
                {
                    return Cancelled;
                }
                WritePng(host.Dest, Required(options, "output"));
                string paramsOut;
                if (options.TryGetValue("params-out", out paramsOut) && host.FilterParameters != null)
                {
                    using (var stream = File.Create(paramsOut))
                    {
                        new BinaryFormatter().Serialize(stream, host.FilterParameters);
                    }
                }
            }
            return Applied;
        }

        private static string Required(Dictionary<string, string> options, string name)
        {
            string value;
            if (!options.TryGetValue(name, out value))
            {
                throw new ArgumentException("--" + name + " is required.");
            }
            return value;
        }

        private static BitmapSource ReadPng(string path)
        {
            using (var stream = File.OpenRead(path))
            {
                BitmapFrame frame = BitmapDecoder.Create(stream, BitmapCreateOptions.PreservePixelFormat, BitmapCacheOption.OnLoad).Frames[0];
                var converted = new FormatConvertedBitmap(frame, PixelFormats.Bgra32, null, 0);
                converted.Freeze();
                return converted;
            }
        }

        private static void WritePng(BitmapSource image, string path)
        {
            var encoder = new PngBitmapEncoder();
            encoder.Frames.Add(BitmapFrame.Create(new FormatConvertedBitmap(image, PixelFormats.Bgra32, null, 0)));
            using (var stream = File.Create(path))
            {
                encoder.Save(stream);
            }
        }

        // Saved filter settings come from recipes and projects, which people share: never let them
        // name arbitrary .NET types (BinaryFormatter would construct them). Only PSFilterHost's own
        // types and plain data are allowed.
        private static ParameterData ReadParameters(string path)
        {
            using (var stream = File.OpenRead(path))
            {
                var formatter = new BinaryFormatter { Binder = new AllowListBinder() };
                return (ParameterData)formatter.Deserialize(stream);
            }
        }

        private sealed class AllowListBinder : SerializationBinder
        {
            private static readonly HashSet<string> PlainTypes = new HashSet<string>(StringComparer.Ordinal)
            {
                "System.Boolean", "System.Byte", "System.SByte", "System.Char", "System.Int16", "System.UInt16",
                "System.Int32", "System.UInt32", "System.Int64", "System.UInt64", "System.Single", "System.Double",
                "System.String", "System.Byte[]", "System.Object[]", "System.Int32[]", "System.UInt32[]", "System.Double[]",
                "System.Collections.Generic.Dictionary`2", "System.Collections.Generic.List`1",
                "System.Collections.Generic.KeyValuePair`2", "System.Collections.Generic.GenericEqualityComparer`1",
                "System.Collections.Generic.ObjectEqualityComparer`1", "System.Collections.Generic.EnumEqualityComparer`1",
                "System.Collections.ObjectModel.ReadOnlyCollection`1",
            };

            public override Type BindToType(string assemblyName, string typeName)
            {
                if (!Allowed(assemblyName, typeName))
                {
                    throw new SerializationException("Saved filter settings name a type that isn't allowed: " + typeName);
                }
                return Type.GetType(typeName + ", " + assemblyName, true);
            }

            private static bool Allowed(string assemblyName, string typeName)
            {
                if (assemblyName.StartsWith("PSFilterHost,", StringComparison.Ordinal) && typeName.StartsWith("PSFilterHostDll.", StringComparison.Ordinal))
                {
                    return GenericArgumentsAllowed(typeName);
                }
                if (!assemblyName.StartsWith("mscorlib,", StringComparison.Ordinal))
                {
                    return false;
                }
                int generic = typeName.IndexOf('[');
                string baseName = generic > 0 && !typeName.EndsWith("[]", StringComparison.Ordinal) ? typeName.Substring(0, generic) : typeName;
                return PlainTypes.Contains(baseName) && GenericArgumentsAllowed(typeName);
            }

            // Generic arguments appear as [[Type, Assembly, ...],[...]]: each must be allowed too.
            private static bool GenericArgumentsAllowed(string typeName)
            {
                int start = typeName.IndexOf("[[", StringComparison.Ordinal);
                if (start < 0)
                {
                    return true;
                }
                foreach (string argument in SplitArguments(typeName.Substring(start + 1, typeName.Length - start - 2)))
                {
                    int comma = TopLevelComma(argument);
                    if (comma < 0 || !Allowed(argument.Substring(comma + 1).Trim(), argument.Substring(0, comma).Trim()))
                    {
                        return false;
                    }
                }
                return true;
            }

            private static IEnumerable<string> SplitArguments(string list)
            {
                int depth = 0, begin = -1;
                for (int i = 0; i < list.Length; i++)
                {
                    if (list[i] == '[')
                    {
                        if (depth++ == 0)
                        {
                            begin = i + 1;
                        }
                    }
                    else if (list[i] == ']' && --depth == 0)
                    {
                        yield return list.Substring(begin, i - begin);
                    }
                }
            }

            private static int TopLevelComma(string text)
            {
                int depth = 0;
                for (int i = 0; i < text.Length; i++)
                {
                    if (text[i] == '[')
                    {
                        depth++;
                    }
                    else if (text[i] == ']')
                    {
                        depth--;
                    }
                    else if (text[i] == ',' && depth == 0)
                    {
                        return i;
                    }
                }
                return -1;
            }
        }

        private static string Json(string value)
        {
            if (value == null)
            {
                return "null";
            }
            var text = new StringBuilder("\"");
            foreach (char c in value)
            {
                switch (c)
                {
                    case '"': text.Append("\\\""); break;
                    case '\\': text.Append("\\\\"); break;
                    default:
                        if (c < 0x20)
                        {
                            text.Append("\\u").Append(((int)c).ToString("x4", CultureInfo.InvariantCulture));
                        }
                        else
                        {
                            text.Append(c);
                        }
                        break;
                }
            }
            return text.Append('"').ToString();
        }
    }
}

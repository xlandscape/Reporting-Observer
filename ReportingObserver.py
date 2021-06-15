"""
A observer running Python reporting tools.
"""

import os
import base
import xml.etree.ElementTree
import glob
import stores
import numpy as np
import datetime
import h5py
import ogr
import shutil


class ReportingObserver(base.Observer):
    """
    A observer that runs Python reporting tools.
    """
    # RELEASES
    VERSION = base.VersionCollection(
        base.VersionInfo("2.0.3", "2021-02-22"),
        base.VersionInfo("2.0.2", "2021-01-28"),
        base.VersionInfo("2.0.1", "2020-12-03"),
        base.VersionInfo("2.0.0", "2020-10-22"),
        base.VersionInfo("1.3.35", "2020-08-12"),
        base.VersionInfo("1.3.29", "2020-06-15"),
        base.VersionInfo("1.3.24", "2020-04-02"),
        base.VersionInfo("1.3.20", "2020-03-23"),
        base.VersionInfo("1.3.19", "2020-03-19"),
        base.VersionInfo("1.3.18", "2020-03-12")
    )

    # CHANGELOG
    VERSION.added("1.3.18", "observer.ReportingObserver")
    VERSION.changed("1.3.19", "Updated observer.ReportingObserver module to v02")
    VERSION.fixed("1.3.20", "observer.ReportingObserver skips processing if all efate inputs are switched off")
    VERSION.added("1.3.24", "observer.ReportingObserver.flush() and observer.ReportingObserver.write()")
    VERSION.changed("1.3.24", "observer.ReportingObserver uses base function to call observer module")
    VERSION.fixed("1.3.24", "observer.ReportingObserver input slicing")
    VERSION.fixed("1.3.29", "observer.ReportingObserver input slicing (again)")
    VERSION.changed("1.3.35", "observer.ReportingObserver receives output folder as environment variable")
    VERSION.changed("2.0.0", "First independent release")
    VERSION.added("2.0.1", "Changelog and release history")
    VERSION.changed("2.0.2", "can handle yearly survival probabilities")
    VERSION.fixed("2.0.3", "survival reporting for Cascade")

    def __init__(self, data, output_folder, **keywords):
        super(ReportingObserver, self).__init__()
        self._output_folder = output_folder
        self._componentPath = os.path.dirname(__file__)
        script = os.path.join(self._componentPath, "module", "bin", "reporting.py")
        if not os.path.isfile(script):
            raise FileNotFoundError('ReportingObserver script "' + script + '" not found')
        # noinspection SpellCheckingInspection
        self._call = [os.path.join(self._componentPath, "module", "bin", "Python", "python.exe"), script, "--fpath",
                      output_folder, "--zip", "false"]
        self._data = data
        self._params = keywords
        return

    def experiment_finished(self, detail=None):
        """
        Function that is called when an experiment has finished.
        :param detail: Additional information.
        :return: Nothing.
        """
        # noinspection SpellCheckingInspection
        if self._params["cascade"] != "true" and self._params["cmfcont"] != "true" and self._params["steps"] != "true":
            self.write_skip_message(os.path.join(self._output_folder, "message.txt"), "no efate data available")
        elif self._params["lguts"] != "true":
            self.write_skip_message(os.path.join(self._output_folder, "message.txt"), "no effect data available")
        else:
            self.prepare_parameterization(os.path.join(self._output_folder, "attributes.xml"))
            # noinspection SpellCheckingInspection
            self.prepare_spray_drift_list(os.path.join(self._output_folder, "SpraydriftList.csv"))
            reaches = self.prepare_reach_list_and_get_reaches(os.path.join(self._output_folder, "ReachList.csv"),
                                                              os.path.join(self._output_folder, "reaches_lguts.csv"))
            self.prepare_efate_effects(os.path.join(self._output_folder, "res.h5"), reaches)
            self.prepare_catchment_list(os.path.join(self._output_folder, "CatchmentList.csv"))
            # noinspection SpellCheckingInspection
            base.run_process(self._call, None, self.default_observer, {"HOMEPATH": self._output_folder})
        return

    def input_get_values(self, component_input):
        """
        Function that is called when values are retrieved.
        :param component_input: The input that delivers values.
        :return: Nothing.
        """
        return

    def mc_run_finished(self, detail=None):
        """
        Function that is called when a Monte Carlo run has finished.
        :param detail: Additional information.
        :return: Nothing.
        """
        return

    def store_set_values(self, level, store_name, message):
        """
        Function that is called when values are saved in a data store.
        :param level: The level of the message.
        :param store_name: The name of the data store.
        :param message: The message itself.
        :return: Nothing.
        """
        return

    def write_message(self, level, message, detail=None):
        """
        Sends a generic message to the observer.
        :param level: The level of the message.
        :param message: The message itself.
        :param detail: Additional information.
        :return: Nothing.
        """
        return

    def mc_run_started(self, composition):
        """
        Function that is called when a Monte Carlo run starts.
        :param composition: The composition of the Monte Carlo run.
        :return: Nothing.
        """
        return

    def prepare_parameterization(self, output_file):
        """
        Prepares the parameterization of the reporting observer module.
        :param output_file: The file path of the parameterization file.
        :return: Nothing.
        """
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        attributes = xml.etree.ElementTree.Element("Attributes")
        for key, value in self._params.items():
            if key != "lock" and key[:3] != "lm_":
                xml.etree.ElementTree.SubElement(attributes, key).text = value
        xml_tree = xml.etree.ElementTree.ElementTree(attributes)
        xml_tree.write(output_file, encoding="utf-8", xml_declaration=True)
        return

    def prepare_spray_drift_list(self, output_file):
        """
        Prepares the spray-drift input for the reporting observer.
        :param output_file: The file path of the spray-drift file.
        :return: Nothing.
        """
        databases = glob.glob(os.path.join(self._data, "mcs", "**", "store"), recursive=True)
        start_date = datetime.datetime.strptime(self._params["lm_simulation_start"] + " 12", "%Y-%m-%d %H")
        with open(output_file, "w") as f:
            f.write("mc,key,substance,time,rate\n")
            for source in databases:
                mc = os.path.normpath(source).split(os.path.sep)[6]
                x3df = stores.X3dfStore(source, mode="r")
                n_days, n_reaches = x3df.describe(self._params["lm_spray_drift_ds"])["shape"]
                reaches = x3df.get_values(self._params["lm_spray_drift_reaches_ds"])
                for r in range(n_reaches):
                    exposure = x3df.get_values(self._params["lm_spray_drift_ds"], slices=(slice(n_days), r))
                    for event in np.nonzero(exposure > 0):
                        if event.shape[0] > 0:
                            f.write("{},".format(mc))
                            f.write("r{},".format(reaches[r]))
                            f.write("{},".format(self._params["lm_compound_name"]))
                            f.write("{},".format((start_date + datetime.timedelta(int(event[0]))).isoformat(" ")))
                            f.write("{:.4f}\n".format(exposure[event][0]))
        return

    def prepare_efate_effects(self, output_file, reaches):
        """
        Prepares the efate and effect inputs of the reporting observer module.
        :param output_file: The file path of the H5 file with the input data.
        :param reaches: A list of reaches.
        :return: Nothing.
        """
        databases = glob.glob(os.path.join(self._data, "mcs", "**", "store"), recursive=True)
        start_time = datetime.datetime.strptime(self._params["t0"], "%Y-%m-%dT%H:%M")
        year_start = datetime.datetime(start_time.year, 1, 1)
        day_difference = int((start_time - year_start).total_seconds() / 3600 / 24)
        with h5py.File(output_file, "w") as f:
            for mc, source in enumerate(databases):
                x3df = stores.X3dfStore(source, mode="r")
                if self._params["cascade"] == "true":
                    n_hours, n_reaches = x3df.describe(self._params["lm_cascade_ds"])["shape"]
                    if self._params["lguts"] == "true":
                        n_days = int(n_hours / 24)
                        offset = day_difference if self._params["lm_cascade_survival_offset"] == "true" else 0
                    cascade_reaches = x3df.get_values(self._params["lm_cascade_reaches"])
                    if mc == 0:
                        # noinspection SpellCheckingInspection
                        f.create_dataset(self._params["cascade_pecsw"], (len(databases), n_reaches, n_hours), np.float,
                                         compression="gzip", chunks=(1, 1, min(n_hours, 2 ** 18)))
                        if self._params["lguts"] == "true":
                            f.create_dataset(self._params["cascade_survival"], (len(databases), n_reaches, n_days),
                                             np.float,
                                             compression="gzip", chunks=(1, 1, min(n_days, 2 ** 18)))
                    for i, reach in enumerate(reaches):
                        r = np.nonzero(cascade_reaches == reach)[0][0]
                        # noinspection SpellCheckingInspection
                        f[self._params["cascade_pecsw"]][mc, i, 0:n_hours] = x3df.get_values(
                            self._params["lm_cascade_ds"], slices=(slice(n_hours), r))
                        if self._params["lguts"] == "true":
                            scales = x3df.describe(self._params["lm_steps_survival"])["scales"]
                            if scales == "time/day, space/base_geometry, other/factor":
                                f[self._params["cascade_survival"]][mc, i, 0:n_days] = x3df.get_values(
                                    self._params["lm_cascade_survival"],
                                    slices=(
                                        slice(offset, offset + n_days),
                                        r,
                                        int(self._params["lm_cascade_survival_mfs_index"])
                                    )
                                )
                            elif scales == "time/year, space/base_geometry, other/factor":
                                for day in range(n_days):
                                    f[self._params["cascade_survival"]][mc, i, day] = x3df.get_values(
                                        self._params["lm_cascade_survival"],
                                        slices=(
                                            (start_time + datetime.timedelta(day)).year - year_start.year,
                                            r,
                                            int(self._params["lm_cascade_survival_mfs_index"])
                                        )
                                    )
                            else:
                                raise ValueError("Unsupported scales: " + scales)
                # noinspection SpellCheckingInspection
                if self._params["cmfcont"] == "true":
                    n_hours, n_reaches = x3df.describe(self._params["lm_cmf_continuous_ds"])["shape"]
                    if self._params["lguts"] == "true":
                        n_days = int(n_hours / 24)
                        offset = day_difference if self._params["lm_cmf_survival_offset"] == "true" else 0
                    cmf_reaches = x3df.get_values(self._params["lm_cmf_continuous_reaches"])
                    if mc == 0:
                        # noinspection SpellCheckingInspection
                        f.create_dataset(self._params["cmfcont_pecsw"], (len(databases), n_reaches, n_hours), np.float,
                                         compression="gzip", chunks=(1, 1, min(n_hours, 2 ** 18)))
                        if self._params["lguts"] == "true":
                            # noinspection SpellCheckingInspection
                            f.create_dataset(self._params["cmfcont_survival"], (len(databases), n_reaches, n_days),
                                             np.float,
                                             compression="gzip", chunks=(1, 1, min(n_days, 2 ** 18)))
                    for i, reach in enumerate(reaches):
                        r = np.nonzero(cmf_reaches == reach)[0][0]
                        # noinspection SpellCheckingInspection
                        f[self._params["cmfcont_pecsw"]][mc, i, 0:n_hours] = x3df.get_values(
                            self._params["lm_cmf_continuous_ds"], slices=(slice(n_hours), r))
                        if self._params["lguts"] == "true":
                            scales = x3df.describe(self._params["lm_steps_survival"])["scales"]
                            if scales == "time/day, space/base_geometry, other/factor":
                                # noinspection SpellCheckingInspection
                                f[self._params["cmfcont_survival"]][mc, i, 0:n_days] = x3df.get_values(
                                    self._params["lm_cmf_continuous_survival"],
                                    slices=(
                                        slice(offset, offset + n_days),
                                        r,
                                        int(self._params["lm_cmf_continuous_survival_mfs_index"])
                                    )
                                )
                            elif scales == "time/year, space/base_geometry, other/factor":
                                for day in range(n_days):
                                    # noinspection SpellCheckingInspection
                                    f[self._params["cmfcont_survival"]][mc, i, day] = x3df.get_values(
                                        self._params["lm_cmf_continuous_survival"],
                                        slices=(
                                            (start_time + datetime.timedelta(day)).year - year_start.year,
                                            r,
                                            int(self._params["lm_cmf_continuous_survival_mfs_index"])
                                        )
                                    )
                            else:
                                raise ValueError("Unsupported scales: " + scales)
                if self._params["steps"] == "true":
                    n_hours, n_reaches = x3df.describe(self._params["lm_steps_rivernetwork_ds"])["shape"]
                    if self._params["lguts"] == "true":
                        n_days = int(n_hours / 24)
                        offset = day_difference if self._params["lm_steps_survival_offset"] == "true" else 0
                    steps_reaches = x3df.get_values(self._params["lm_steps_rivernetwork_reaches"])
                    if mc == 0:
                        # noinspection SpellCheckingInspection
                        f.create_dataset(self._params["steps_pecsw"], (len(databases), n_reaches, n_hours), np.float,
                                         compression="gzip", chunks=(1, 1, min(n_hours, 2 ** 18)))
                        if self._params["lguts"] == "true":
                            f.create_dataset(self._params["steps_survival"], (len(databases), n_reaches, n_days),
                                             np.float,
                                             compression="gzip", chunks=(1, 1, min(n_days, 2 ** 18)))
                    for i, reach in enumerate(reaches):
                        r = np.nonzero(steps_reaches == reach)[0][0]
                        # noinspection SpellCheckingInspection
                        f[self._params["steps_pecsw"]][mc, i, 0:n_hours] = x3df.get_values(
                            self._params["lm_steps_rivernetwork_ds"], slices=(slice(n_hours), r))
                        if self._params["lguts"] == "true":
                            scales = x3df.describe(self._params["lm_steps_survival"])["scales"]
                            if scales == "time/day, space/base_geometry, other/factor":
                                f[self._params["steps_survival"]][mc, i, 0:n_days] = x3df.get_values(
                                    self._params["lm_steps_survival"],
                                    slices=(
                                        slice(offset, offset + n_days),
                                        r,
                                        int(self._params["lm_steps_survival_mfs_index"])
                                    )
                                )
                            elif scales == "time/year, space/base_geometry, other/factor":
                                for day in range(n_days):
                                    f[self._params["steps_survival"]][mc, i, day] = x3df.get_values(
                                        self._params["lm_steps_survival"],
                                        slices=(
                                            (start_time + datetime.timedelta(day)).year - year_start.year,
                                            r,
                                            int(self._params["lm_steps_survival_mfs_index"])
                                        )
                                    )
                            else:
                                raise ValueError("Unsupported scales: " + scales)
                if mc == 0:
                    f.create_dataset(self._params["cmf_depth"], (len(databases), n_reaches, n_hours), np.float,
                                     compression="gzip", chunks=(1, 1, min(n_hours, 2 ** 18)))
                for i, reach in enumerate(reaches):
                    cmf_reaches = x3df.get_values(self._params["lm_cmf_reaches_ds"])
                    r = np.nonzero(cmf_reaches == reach)[0][0]
                    f[self._params["cmf_depth"]][mc, i, 0:n_hours] = x3df.get_values(self._params["lm_cmf_depth_ds"],
                                                                                     slices=(slice(n_hours), r))
        return

    def prepare_reach_list_and_get_reaches(self, output_file, output_file2):
        """
        Prepares the reach input of the reporting module.
        :param output_file: The path of the first reaches input file.
        :param output_file2: The path of the second reaches input file.
        :return: Nothing
        """
        driver = ogr.GetDriverByName("ESRI Shapefile")
        databases = glob.glob(os.path.join(self._data, "mcs", "**", "store"), recursive=True)
        x3df = stores.X3dfStore(databases[0], mode="r")
        hydrography = x3df.get_values(self._params["lm_hydrography_ds"])
        data_source = driver.Open(hydrography, 0)
        layer = data_source.GetLayer()
        reaches = np.zeros((layer.GetFeatureCount(),), np.int)
        with open(output_file, "w") as f:
            with open(output_file2, "w") as f2:
                f.write("key,x,y,downstream\n")
                for index, feature in enumerate(layer):
                    reaches[index] = feature.GetField("key")
                    geom = feature.GetGeometryRef()
                    coord = geom.GetPoint(0)
                    downstream = feature.GetField("downstream")
                    f.write("r{},".format(reaches[index]))
                    f.write("{:.2f},".format(coord[0]))
                    f.write("{:.2f},".format(coord[1]))
                    f.write("{}{}\n".format("" if downstream == "Outlet" else "r", downstream))
                    f2.write("      r{}\n".format(reaches[index]))
        return reaches

    def prepare_catchment_list(self, output_file):
        """
        Prepares the catchment input of the reporting observer module.
        :param output_file: The file path of the catchment input.
        :return: Nothing.
        """
        shutil.copyfile(self._params["lm_catchment"], output_file)
        return

    @staticmethod
    def write_skip_message(output_file, reason):
        """
        Writes a message that indicates why no reporting was conducted.
        :param output_file: The file path of the message file.
        :param reason: The reason why the reporting observer did not run.
        :return: Nothing
        """
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            f.write("Skipped reporting\nReason: {}\n".format(reason))
        return

    def flush(self):
        """
        Flushes the buffer of the reporter.
        :return: Nothing.
        """
        return

    def write(self, text):
        """
        Requests the reporter to write text.
        :param text: The text to write.
        :return: Nothing.
        """
        return

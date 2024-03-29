"""An observer running Python reporting tools."""

import os
import base
import xml.etree.ElementTree
import glob
import stores
import numpy as np
import datetime
import h5py
from osgeo import ogr
import shutil


class ReportingObserver(base.Observer):
    """An observer that runs Python reporting tools."""
    # RELEASES
    VERSION = base.VersionCollection(
        base.VersionInfo("2.0.15", "2023-09-13"),
        base.VersionInfo("2.0.14", "2023-09-12"),
        base.VersionInfo("2.0.13", "2023-09-11"),
        base.VersionInfo("2.0.12", "2023-03-09"),
        base.VersionInfo("2.0.11", "2021-11-18"),
        base.VersionInfo("2.0.10", "2021-10-19"),
        base.VersionInfo("2.0.9", "2021-10-12"),
        base.VersionInfo("2.0.8", "2021-10-11"),
        base.VersionInfo("2.0.7", "2021-09-02"),
        base.VersionInfo("2.0.6", "2021-08-25"),
        base.VersionInfo("2.0.5", "2021-08-16"),
        base.VersionInfo("2.0.4", "2021-07-19"),
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
    VERSION.added("1.3.18", "`observer.ReportingObserver` ")
    VERSION.changed("1.3.19", "Updated `observer.ReportingObserver` module to v02")
    VERSION.fixed("1.3.20", "`observer.ReportingObserver` skips processing if all fate inputs are switched off")
    VERSION.added("1.3.24", "`observer.ReportingObserver.flush()` and `observer.ReportingObserver.write()`")
    VERSION.changed("1.3.24", "`observer.ReportingObserver` uses base function to call observer module")
    VERSION.fixed("1.3.24", "`observer.ReportingObserver` input slicing")
    VERSION.fixed("1.3.29", "`observer.ReportingObserver` input slicing (again)")
    VERSION.changed("1.3.35", "`observer.ReportingObserver` receives output folder as environment variable")
    VERSION.changed("2.0.0", "First independent release")
    VERSION.added("2.0.1", "Changelog and release history")
    VERSION.changed("2.0.2", "Can handle yearly survival probabilities")
    VERSION.fixed("2.0.3", "Survival reporting for Cascade")
    VERSION.changed("2.0.4", "Changelog uses markdown")
    VERSION.changed("2.0.4", "Spellings")
    VERSION.fixed("2.0.5", "Referencing of wrong datasets when obtaining scales")
    VERSION.added("2.0.6", "Base documentation")
    VERSION.added("2.0.7", "ogr module import")
    VERSION.changed("2.0.8", "Replaced legacy format strings by f-strings")
    VERSION.changed("2.0.9", "Switched to Google docstring style")
    VERSION.changed("2.0.10", "Specified working directory for module")
    VERSION.changed("2.0.11", "Removed reaches inputs")
    VERSION.changed("2.0.12", "Removed sample project from ReportingObserver module due to file size restrictions")
    VERSION.added("2.0.13", "Information on runtime environment")
    VERSION.changed("2.0.14", "Extended module information for Python runtime environment")
    VERSION.added("2.0.14", "Creation of repository info during documentation")
    VERSION.added("2.0.14", "Repository info, changelog, contributing note and license to module")
    VERSION.added("2.0.14", "Repository info to Python runtime environment")
    VERSION.fixed("2.0.15", "Scales of inputs")

    MODULE = base.Module(
        "create reporting aqRisk@LandcapeModel",
        "0.2",
        "module",
        None,
        base.Module(
            "Python",
            "3.7.4",
            "module/bin/python",
            "module/bin/python/Doc/python374.chm",
            None,
            True,
            "module/bin/python/NEWS.txt"
        )
    )

    def __init__(self, data, output_folder, **keywords):
        """
        Initializes a ReportingObserver.

        Args:
            data: The input data of the observer.
            output_folder: The folder for the output of the reporter
            **keywords: Additional keywords.
        """
        super(ReportingObserver, self).__init__()
        self._output_folder = output_folder
        self._componentPath = os.path.dirname(__file__)
        script = os.path.join(self._componentPath, "module", "bin", "reporting.py")
        if not os.path.isfile(script):
            raise FileNotFoundError(f"ReportingObserver script '{script}' not found")
        # noinspection SpellCheckingInspection
        self._call = [os.path.join(self._componentPath, "module", "bin", "Python", "python.exe"), script, "--fpath",
                      output_folder, "--zip", "false"]
        self._data = data
        self._params = keywords

    def experiment_finished(self, detail=None):
        """
        Reacts when an experiment is completed.

        Args:
            detail: Additional details to report.

        Returns:
             Nothing.
        """
        # noinspection SpellCheckingInspection
        if self._params["cascade"] != "true" and self._params["cmfcont"] != "true" and self._params["steps"] != "true":
            self.write_skip_message(os.path.join(self._output_folder, "message.txt"), "no fate data available")
        elif self._params["lguts"] != "true":
            self.write_skip_message(os.path.join(self._output_folder, "message.txt"), "no effect data available")
        else:
            self.prepare_parameterization(os.path.join(self._output_folder, "attributes.xml"))
            # noinspection SpellCheckingInspection
            self.prepare_spray_drift_list(os.path.join(self._output_folder, "SpraydriftList.csv"))
            # noinspection SpellCheckingInspection
            reaches = self.prepare_reach_list_and_get_reaches(os.path.join(self._output_folder, "ReachList.csv"),
                                                              os.path.join(self._output_folder, "reaches_lguts.csv"))
            self.prepare_fate_and_effects(os.path.join(self._output_folder, "res.h5"), reaches)
            self.prepare_catchment_list(os.path.join(self._output_folder, "CatchmentList.csv"))
            # noinspection SpellCheckingInspection
            base.run_process(self._call, self._output_folder, self.default_observer, {"HOMEPATH": self._output_folder})

    def prepare_parameterization(self, output_file):
        """
        Prepares the parameterization of the reporting observer module.

        Args:
            output_file: The file path of the parameterization file.

        Returns:
            Nothing.
        """
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        attributes = xml.etree.ElementTree.Element("Attributes")
        for key, value in self._params.items():
            if key != "lock" and key[:3] != "lm_":
                xml.etree.ElementTree.SubElement(attributes, key).text = value
        xml_tree = xml.etree.ElementTree.ElementTree(attributes)
        xml_tree.write(output_file, encoding="utf-8", xml_declaration=True)

    def prepare_spray_drift_list(self, output_file):
        """
        Prepares the spray-drift input for the reporting observer.

        Args:
            output_file: The file path of the spray-drift file.

        Returns:
            Nothing.
        """
        databases = glob.glob(os.path.join(self._data, "mcs", "**", "store"), recursive=True)
        start_date = datetime.datetime.strptime(f"{self._params['lm_simulation_start']} 12", "%Y-%m-%d %H")
        with open(output_file, "w") as f:
            f.write("mc,key,substance,time,rate\n")
            for source in databases:
                mc = os.path.normpath(source).split(os.path.sep)[6]
                x3df = stores.X3dfStore(source, mode="r")
                n_days, n_reaches = x3df.describe(self._params["lm_spray_drift_ds"])["shape"]
                reaches = x3df.describe(self._params["lm_spray_drift_ds"])["element_names"][1].get_values()
                for r in range(n_reaches):
                    exposure = x3df.get_values(self._params["lm_spray_drift_ds"], slices=(slice(n_days), r))
                    for event in np.nonzero(exposure > 0):
                        if event.shape[0] > 0:
                            f.write(f"{mc},")
                            f.write(f"r{reaches[r]},")
                            f.write(f"{self._params['lm_compound_name']},")
                            f.write(f"{(start_date + datetime.timedelta(int(event[0]))).isoformat(' ')},")
                            f.write(f"{format(exposure[event][0], '4f')}\n")

    def prepare_fate_and_effects(self, output_file, reaches):
        """
        Prepares the environmental fate and effect inputs of the reporting observer module.

        Args:
            output_file: The file path of the H5 file with the input data.
            reaches: A list of reaches.

        Returns:
            Nothing.
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
                    # noinspection SpellCheckingInspection
                    if self._params["lguts"] == "true":
                        n_days = int(n_hours / 24)
                        offset = day_difference if self._params["lm_cascade_survival_offset"] == "true" else 0
                    cascade_reaches = x3df.describe(self._params["lm_cascade_ds"])["element_names"][1].get_values()
                    if mc == 0:
                        # noinspection SpellCheckingInspection
                        f.create_dataset(self._params["cascade_pecsw"], (len(databases), n_reaches, n_hours), np.float,
                                         compression="gzip", chunks=(1, 1, min(n_hours, 2 ** 18)))
                        # noinspection SpellCheckingInspection
                        if self._params["lguts"] == "true":
                            f.create_dataset(self._params["cascade_survival"], (len(databases), n_reaches, n_days),
                                             np.float,
                                             compression="gzip", chunks=(1, 1, min(n_days, 2 ** 18)))
                    for i, reach in enumerate(reaches):
                        r = np.nonzero(cascade_reaches == reach)[0][0]
                        # noinspection SpellCheckingInspection
                        f[self._params["cascade_pecsw"]][mc, i, 0:n_hours] = x3df.get_values(
                            self._params["lm_cascade_ds"], slices=(slice(n_hours), r))
                        # noinspection SpellCheckingInspection
                        if self._params["lguts"] == "true":
                            scales = x3df.describe(self._params["lm_cascade_survival"])["scales"]
                            if scales == "time/day, space/reach, other/factor":
                                f[self._params["cascade_survival"]][mc, i, 0:n_days] = x3df.get_values(
                                    self._params["lm_cascade_survival"],
                                    slices=(
                                        slice(offset, offset + n_days),
                                        r,
                                        int(self._params["lm_cascade_survival_mfs_index"])
                                    )
                                )
                            elif scales == "time/year, space/reach, other/factor":
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
                                raise ValueError(f"Unsupported scales: {scales}")
                # noinspection SpellCheckingInspection
                if self._params["cmfcont"] == "true":
                    n_hours, n_reaches = x3df.describe(self._params["lm_cmf_continuous_ds"])["shape"]
                    # noinspection SpellCheckingInspection
                    if self._params["lguts"] == "true":
                        n_days = int(n_hours / 24)
                        offset = day_difference if self._params["lm_cmf_survival_offset"] == "true" else 0
                    cmf_reaches = x3df.describe(self._params["lm_cmf_continuous_ds"])["element_names"][1].get_values()
                    if mc == 0:
                        # noinspection SpellCheckingInspection
                        f.create_dataset(
                            self._params["cmfcont_pecsw"],
                            (len(databases), n_reaches, n_hours),
                            np.float,
                            compression="gzip",
                            chunks=(1, 1, min(n_hours, 2 ** 18))
                        )
                        # noinspection SpellCheckingInspection
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
                        # noinspection SpellCheckingInspection
                        if self._params["lguts"] == "true":
                            scales = x3df.describe(self._params["lm_cmf_continuous_survival"])["scales"]
                            if scales == "time/day, space/reach, other/factor":
                                # noinspection SpellCheckingInspection
                                f[self._params["cmfcont_survival"]][mc, i, 0:n_days] = x3df.get_values(
                                    self._params["lm_cmf_continuous_survival"],
                                    slices=(
                                        slice(offset, offset + n_days),
                                        r,
                                        int(self._params["lm_cmf_continuous_survival_mfs_index"])
                                    )
                                )
                            elif scales == "time/year, space/reach, other/factor":
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
                                raise ValueError(f"Unsupported scales: {scales}")
                if self._params["steps"] == "true":
                    n_hours, n_reaches = x3df.describe(self._params["lm_steps_river_network_ds"])["shape"]
                    # noinspection SpellCheckingInspection
                    if self._params["lguts"] == "true":
                        n_days = int(n_hours / 24)
                        offset = day_difference if self._params["lm_steps_survival_offset"] == "true" else 0
                    steps_reaches = x3df.describe(
                        self._params["lm_steps_river_network_ds"])["element_names"][1].get_values()
                    if mc == 0:
                        # noinspection SpellCheckingInspection
                        f.create_dataset(self._params["steps_pecsw"], (len(databases), n_reaches, n_hours), np.float,
                                         compression="gzip", chunks=(1, 1, min(n_hours, 2 ** 18)))
                        # noinspection SpellCheckingInspection
                        if self._params["lguts"] == "true":
                            f.create_dataset(self._params["steps_survival"], (len(databases), n_reaches, n_days),
                                             np.float,
                                             compression="gzip", chunks=(1, 1, min(n_days, 2 ** 18)))
                    for i, reach in enumerate(reaches):
                        r = np.nonzero(steps_reaches == reach)[0][0]
                        # noinspection SpellCheckingInspection
                        f[self._params["steps_pecsw"]][mc, i, 0:n_hours] = x3df.get_values(
                            self._params["lm_steps_river_network_ds"], slices=(slice(n_hours), r))
                        # noinspection SpellCheckingInspection
                        if self._params["lguts"] == "true":
                            scales = x3df.describe(self._params["lm_steps_survival"])["scales"]
                            if scales == "time/day, space/reach, other/factor":
                                f[self._params["steps_survival"]][mc, i, 0:n_days] = x3df.get_values(
                                    self._params["lm_steps_survival"],
                                    slices=(
                                        slice(offset, offset + n_days),
                                        r,
                                        int(self._params["lm_steps_survival_mfs_index"])
                                    )
                                )
                            elif scales == "time/year, space/reach, other/factor":
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
                                raise ValueError(f"Unsupported scales: {scales}")
                if mc == 0:
                    f.create_dataset(self._params["cmf_depth"], (len(databases), n_reaches, n_hours), np.float,
                                     compression="gzip", chunks=(1, 1, min(n_hours, 2 ** 18)))
                for i, reach in enumerate(reaches):
                    cmf_reaches = x3df.describe(self._params["lm_cmf_depth_ds"])["element_names"][1].get_values()
                    r = np.nonzero(cmf_reaches == reach)[0][0]
                    f[self._params["cmf_depth"]][mc, i, 0:n_hours] = x3df.get_values(self._params["lm_cmf_depth_ds"],
                                                                                     slices=(slice(n_hours), r))

    def prepare_reach_list_and_get_reaches(self, output_file, output_file2):
        """
        Prepares the reach input of the reporting module.

        Args:
            output_file: The path of the first reaches input file.
            output_file2: The path of the second reaches input file.

        Returns:
            Nothing
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
                    f.write(f"r{reaches[index]},")
                    f.write(f"{format(coord[0], '2f')},")
                    f.write(f"{format(coord[1], '2f')},")
                    f.write(f"{'' if downstream == 'Outlet' else 'r'}{downstream}\n")
                    f2.write(f"      r{reaches[index]}\n")
        return reaches

    def prepare_catchment_list(self, output_file):
        """
        Prepares the catchment input of the reporting observer module.

        Args:
            output_file: The file path of the catchment input.

        Returns:
            Nothing.
        """
        shutil.copyfile(self._params["lm_catchment"], output_file)
        return

    @staticmethod
    def write_skip_message(output_file, reason):
        """
        Writes a message that indicates why no reporting was conducted.

        Args:
            output_file: The file path of the message file.
            reason: The reason why the reporting observer did not run.

        Returns:
            Nothing.
        """
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            f.write(f"Skipped reporting\nReason: {reason}\n")

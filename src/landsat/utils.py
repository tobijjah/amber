import re
import os
import rasterio
from pathlib import Path
from zipfile import ZipFile, is_zipfile
from tarfile import TarFile, is_tarfile
from collections import namedtuple, OrderedDict
from contextlib import contextmanager


BAND_MAP = {
    'LANDSAT_1_MSS': dict(zip('green red nir1 nir2'.split(), '4 5 6 7'.split())),
    'LANDSAT_2_MSS': dict(zip('green red nir1 nir2'.split(), '4 5 6 7'.split())),
    'LANDSAT_3_MSS': dict(zip('green red nir1 nir2'.split(), '4 5 6 7'.split())),
    'LANDSAT_4_MSS': dict(zip('green red nir1 nir2'.split(), '1 2 3 4'.split())),
    'LANDSAT_5_MSS': dict(zip('green red nir1 nir2'.split(), '1 2 3 4'.split())),
    'LANDSAT_4_TM': dict(zip('blue green red nir swir1 tirs swir2'.split(), '1 2 3 4 5 6 7'.split())),
    'LANDSAT_5_TM': dict(zip('blue green red nir swir1 tirs swir2'.split(), '1 2 3 4 5 6 7'.split())),
    'LANDSAT_7_ETM': dict(zip('blue green red nir swir1 tirs_low tirs_high swir2 panchromatic bq'.split(),
                              '1 2 3 4 5 6_VCID_1 6_VCID_2 7 8 QUALITY'.split())),
    'LANDSAT_8_OLI_TIRS': dict(zip('costal blue green red nir swir1 swir2 panchromatic cirrus tirs1 tirs2 bq'.split(),
                                   '1 2 3 4 5 6 7 8 9 10 11 QUALITY')),
}


# Base exception class
class LandsatError(Exception):
    """Base exception"""


# LandsatArchive exception
class LandsatArchiveError(LandsatError):
    """Base class for LandsatArchive exceptions"""


class BandMapError(LandsatArchiveError):
    """Exception for spacecraft + sensor not represented in BAND_MAP"""


class UnsupportedSourceError(LandsatArchiveError):
    """Exception for unsupported sources"""


# LandsatMetadata exceptions
class LandsatMetadataError(LandsatError):
    """Base class for LandsatMetadata exceptions"""


class MetadataFileError(LandsatMetadataError):
    """Exception for Landsat metadata files"""


class ParsingError(LandsatMetadataError):
    """Exception for errors occurring during metadata file parsing"""


class GroupError(LandsatMetadataError):
    """Exception if for a requested but missing group"""


class _TarFileWrapper(TarFile):
    """Delegate a namelist call to TarFile.getnames. Required for convenient duck typing
    between ZipFile and TarFile"""
    def namelist(self):
        return self.getnames()


class LandsatArchive(object):
    def __init__(self, source, metadata_obj, alias, band_mapping):
        self.src = source
        self.alias = alias
        self.metadata = metadata_obj

        self._bands = {}
        self._mapping = band_mapping

    @classmethod
    def read(cls, source, extract_to=None, alias=None, meta_template=r'.*_?MTL.txt', band_mapping=BAND_MAP):
        src = Path(source)

        if src.is_dir():
            return cls.directory_read(src, alias, meta_template, band_mapping)

        elif src.suffix == '.txt' and src.is_file():
            return cls.metadata_read(src, alias, band_mapping)

        elif is_zipfile(str(src)) or is_tarfile(str(src)):
            return cls.archive_read(src, extract_to, alias, meta_template, band_mapping)

        else:
            raise UnsupportedSourceError('%s is not supported.' % source)

    @classmethod
    def directory_read(cls, directory, alias, meta_template, band_mapping):
        meta_file = cls.metadata_sniffer(os.listdir(str(directory)), meta_template)
        meta = LandsatMetadata(directory / meta_file)
        meta.parse()

        mapping = __class__.dispatch_mapping(meta, band_mapping)

        obj = cls(directory, meta, alias, mapping)
        obj._load()

        return obj

    @classmethod
    def metadata_read(cls, metadata, alias, band_mapping):
        meta = LandsatMetadata(metadata)
        meta.parse()

        mapping = __class__.dispatch_mapping(meta, band_mapping)

        path = metadata.parent

        obj = cls(path, meta, alias, mapping)
        obj._load()

        return obj

    @classmethod
    def archive_read(cls, archive, extract_to, alias, meta_template, band_mapping):
        if extract_to is None:
            ex = archive.parent / archive.name.split('.')[0]

        else:
            ex = Path(extract_to)

        with __class__.archive_opener(str(archive)) as src:
            meta_file = cls.metadata_sniffer(src.namelist(), meta_template)
            src.extractall(str(ex))

        meta = LandsatMetadata(ex / meta_file)
        meta.parse()

        mapping = __class__.dispatch_mapping(meta, band_mapping)

        obj = cls(ex, meta, alias, mapping)
        obj._load()

        return obj

    def _load(self):
        regex = re.compile(r'FILE_NAME_BAND_(?P<key>(?:\d{1,2}|[A-Za-z]+).*)', re.I)

        for k, v in self.metadata.iter_group('PRODUCT_METADATA'):
            match = regex.match(k)

            if match:
                self._bands[match.group('key')] = v

    def __getitem__(self, item):
        # TODO not final
        idx = str(item)

        if idx in self._bands:
            return rasterio.open(str(self.src / self._bands[idx]), 'r')

        elif idx in self._mapping:
            idx = self._mapping[idx]
            return rasterio.open(str(self.src / self._bands[idx]), 'r')

        else:
            raise KeyError('%s not found' % item)

    def __str__(self):
        # TODO not final
        msg = """
        Spacecraft: {}
        Sensor: {}
        Date acquired: {}
        Cloud cover: {}
        Quality: {}
        """.format(self.metadata.get('product_metadata', 'spacecraft_id'),
                   self.metadata.get('product_metadata', 'sensor_id'),
                   self.metadata.get('product_metadata', 'date_acquired'),
                   self.metadata.get('image_attributes', 'cloud_cover'),
                   self.metadata.get('image_attributes', 'image_quality'))

        return msg

    def __repr__(self):
        return '{}({}, {}, {}, {})'.format(__class__.__name__, self.src, self.metadata, self.alias, self._mapping)

    @staticmethod
    def dispatch_mapping(meta, band_mapping):
        spacecraft = meta.get('PRODUCT_METADATA', 'SPACECRAFT_ID')
        sensor = meta.get('PRODUCT_METADATA', 'SENSOR_ID')

        if spacecraft is None or sensor is None:
            raise MetadataFileError('Metadata does not contain a spacecraft or sensor attribute')

        mapping = band_mapping.get('%s_%s' % (spacecraft, sensor))

        if mapping is None:
            raise BandMapError('No band mapping found for %s_%s' % (spacecraft, sensor))

        return mapping

    @staticmethod
    def metadata_sniffer(names, template):
        regex = re.compile(template)

        for name in names:
            name = os.path.basename(name)

            if bool(regex.match(name)):
                return name

        raise MetadataFileError('Missing Landsat metadata file in %s' % names)

    @staticmethod
    @contextmanager
    def archive_opener(archive, mode='r'):
        try:
            if is_tarfile(archive):
                opener = _TarFileWrapper.open(archive, mode=mode)
                yield opener

            elif is_zipfile(archive):
                opener = ZipFile(archive, mode=mode)
                yield opener

            else:
                raise UnsupportedSourceError('Unsupported archive file %s' % archive)

        # TODO meaningful error message
        except:
            pass

        finally:
            opener.close()


class LandsatMetadata(object):
    # hook for testing
    _OPENER = open

    def __init__(self, path):
        self.path = Path(path)

    def _asdict(self):
        return OrderedDict([(k, v._asdict())
                            for k, v in self.__dict__.items()
                            if v is not self.path])

    def _delete_attributes(self):
        for attr in self._asdict():
            delattr(self, attr)

    def parse(self):
        metadata = self.__class__.read(str(self.path))

        for item in metadata:
            self.__setattr__(item.GROUP, item)

    def get(self, group, value=None, default=None):
        try:
            attr = self.__getattribute__(group.upper())

            if value is None:
                return attr

            else:
                return attr.__getattribute__(value.upper())

        except AttributeError:
            return default

    def iter_group(self, group):
        attr = self.get(group)

        if attr is None:
            raise GroupError('Not possible to iterate over non existing group: %s' % group)

        yield from attr._asdict().items()

    # TODO return meaningful string currently it is a mess
    def __str__(self):
        return self.__repr__() + '\nMetadata attr: {}'.format(self._asdict())

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.path)

    @staticmethod
    def read(path):
        with __class__._OPENER(path, 'r') as src:
            content = src.readlines()
            scan = __class__.scanner(content)
            lex = __class__.lexer(scan)
            metadata = __class__.parser(lex)

        return metadata

    @staticmethod
    def scanner(content):
        for line in content:
            yield line.strip()

    @staticmethod
    def lexer(scanner):
        start_group = re.compile(r'GROUP\s=\s(?P<start_tag>.+)')
        end_group = re.compile(r'END_GROUP\s=\s(?P<end_tag>.+)')
        eof = 'END'

        groups = []
        for line in scanner:
            if bool(start_group.match(line)):
                current = [line]
                groups.append(current)

            elif bool(end_group.match(line)):
                s_tag = start_group.match(groups[-1][0]).group('start_tag')
                e_tag = end_group.match(line).group('end_tag')

                if s_tag == e_tag:
                    yield groups.pop()
                else:
                    raise ParsingError('Diverging start and end tag: %s != %s' % (s_tag, e_tag))

            elif line == eof:
                return

            else:
                groups[-1].append(line)

    @staticmethod
    def parser(lexer):
        regex = re.compile(r'(?P<key>.+)\s=\s(?P<value>.+)')

        metadata = []
        for group in lexer:
            keys = []
            values = []

            for item in group:
                # skip item if not properly formatted
                try:
                    key, value = regex.match(item).groups()
                except AttributeError:
                    continue

                value = __class__.cast_to_best(value)
                keys.append(key)
                values.append(value)

            if len(keys) == len(values) and len(keys) > 1:
                Metadata = namedtuple('Metadata', keys)
                obj = Metadata(*values)
                metadata.append(obj)

        if len(metadata) == 0:
            raise ParsingError('Empty metadata file')

        return metadata

    @staticmethod
    def cast_to_best(value):
        try:
            return int(value)

        except ValueError:
            try:
                return float(value)

            except ValueError:
                return '{}'.format(value.strip('"'))

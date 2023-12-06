## Code adapted and improved from https://github.com/sebriois/biomart

import os
import pprint
from xml.etree.ElementTree import Element, SubElement, tostring, fromstring
import requests


class BiomartServer(object):
    def __init__(self, url, **kwargs):
        self._databases = {}
        self._datasets = {}

        if not url.startswith('http://'):
            url = 'http://' + url

        self.url = url
        self.http_proxy = kwargs.get('http_proxy', os.environ.get('http_proxy', None))
        self.https_proxy = kwargs.get('https_proxy', os.environ.get('https_proxy', None))
        self._verbose = kwargs.get('verbose', False)

        self.is_alive = False
        self.assert_alive()

    def get_verbose(self):
        return self._verbose

    def set_verbose(self, value):
        if not isinstance(value, bool):
            raise Exception("verbose must be set to a boolean value")

        setattr(self, '_verbose', value)

        # propagate verbose state to databases and datasets objects
        for database in self._databases.values():
            database.verbose = True

        for dataset in self._datasets.values():
            dataset.verbose = True
    verbose = property(get_verbose, set_verbose)
    
    def assert_alive(self):
        if not self.is_alive:
            self.get_request()
            self.is_alive = True
            if self.verbose:
                print("[BiomartServer:'%s'] is alive." % self.url)
    
    @property
    def databases(self):
        if not self._databases:
            self.fetch_databases()
        return self._databases
    
    @property
    def datasets(self):
        if not self._datasets:
            self.fetch_datasets()
        return self._datasets

    def fetch_databases(self):
        if self.verbose:
            print("[BiomartServer:'%s'] Fetching databases" % self.url)

        r = self.get_request(type = 'registry')
        xml = fromstring(r.text)
        for database in xml.findall('MartURLLocation'):
            name = database.attrib['name']

            self._databases[name] = BiomartDatabase(
                server = self,
                name = name,
                display_name = database.attrib['displayName'],
                virtual_schema = database.attrib['serverVirtualSchema'],
                verbose = self._verbose
            )
    
    def fetch_datasets(self):
        if self.verbose:
            print("[BiomartServer:'%s'] Fetching datasets" % self.url)

        for database in self.databases.values():
            self._datasets.update(database.datasets)
    
    def show_databases(self):
        pprint.pprint(self.databases)
    
    def show_datasets(self):
        pprint.pprint(self.datasets)
    
    def get_request(self, **params):
        proxies = {
            'http': self.http_proxy,
            'https': self.https_proxy
        }
        if params:
            r = requests.get(self.url, params = params, proxies = proxies, stream = True)
        else:
            r = requests.get(self.url, proxies = proxies)
        r.raise_for_status()

        return r
    

class BiomartDatabase(object):
    def __init__(self, *args, **kwargs):
        server = kwargs.get('server', None)
        if not server:
            url = args[0]
            server = BiomartServer(url = url, **kwargs)

        self.server = server

        self.name = kwargs.get('name', None)
        if not self.name:
            raise Exception("[BiomartDatabase] 'name' is required")

        self.display_name = kwargs.get('display_name', self.name)
        self.virtual_schema = kwargs.get('virtual_schema', 'default')
        self.verbose = kwargs.get('verbose', False)

        self._datasets = {}
    
    def __repr__(self):
        return self.display_name
    
    @property
    def datasets(self):
        if not self._datasets:
            self.fetch_datasets()
        return self._datasets
    
    def show_datasets(self):
        import pprint
        pprint.pprint(self.datasets)

    def fetch_datasets(self):
        if self.verbose:
            print("[BiomartDatabase:'%s'] Fetching datasets" % self)

        r = self.server.get_request(type = 'datasets', mart = self.name)
        for line in r.iter_lines():
            line = line.decode('utf-8')
            if line:
                cols = line.split("\t")
                if len(cols) > 7:
                    name = cols[1]
                    self._datasets[name] = BiomartDataset(
                        server = self.server,
                        database = self,
                        name = name,
                        display_name = cols[2],
                        interface = cols[7],
                        verbose = self.verbose,
                    )

class BiomartDataset(object):
    def __init__(self, *args, **kwargs):
        # dataset specific attributes
        self.name = kwargs.get('name', None)
        if not self.name:
            raise Exception("[BiomartDataset] 'name' is required")

        self.display_name = kwargs.get('display_name', self.name)
        self.interface = kwargs.get('interface', 'default')
        self.verbose = kwargs.get('verbose', False)

        # get related biomart server
        server = kwargs.get('server', None)
        if not server:
            url = args[0]
            server = BiomartServer(url = url, **kwargs)
        self.server = server

        # get related biomart database
        self.database = kwargs.get('database', None)

        self._filters = {}
        self._attribute_pages = {}

    def __repr__(self):
        return self.display_name

    @property
    def attributes(self):
        """
        A dictionary mapping names of attributes to BiomartAttribute instances.

        This causes overwriting errors if there are diffferent pages which use
        the same attribute names, but is kept for backward compatibility.
        """
        if not self._attribute_pages:
            self.fetch_attributes()
        result = {}
        for page in self._attribute_pages.values():
            result.update(page.attributes)
        return result


    @property
    def attribute_pages(self):
        """
        A dictionary mapping pages of attributes to BiomartAttributePage instances.
        Lists of attributes for particular pages can be accessed by 'attributes'
        field of pages instances.
        """
        if not self._attribute_pages:
            self.fetch_attributes()
        return self._attribute_pages


    @property
    def filters(self):
        if not self._filters:
            self.fetch_filters()
        return self._filters

    def show_filters(self):
        pprint.pprint(self.filters)

    def show_attributes(self):
        pprint.pprint(self.attributes)

    def show_attributes_by_page(self):
        pprint.pprint(self.attribute_pages)

    def fetch_filters(self):
        if self.verbose:
            print("[BiomartDataset:'%s'] Fetching filters" % self.name)

        r = self.server.get_request(type="filters", dataset=self.name)
        for line in r.iter_lines():
            line = line.decode('utf8')
            if line:
                line = line.split("\t")
                self._filters[line[0]] = BiomartFilter(
                    name = line[0],
                    display_name = line[1],
                    accepted_values = line[2],
                    filter_type = line[5],
                )


        # retrieve additional filters from the dataset configuration page
        r = self.server.get_request(type="configuration", dataset=self.name)
        xml = fromstring(r.text)

        for attribute_page in xml.findall('./AttributePage'):

            for attribute in attribute_page.findall('./*/*/AttributeDescription[@pointerFilter]'):
                name = attribute.get('pointerFilter')

                if not name in self._filters:
                    self._filters[name] = BiomartFilter(
                        name = name,
                        display_name = attribute.get('displayName') or name,
                        accepted_values = '',
                        filter_type = '',
                    )

    def fetch_attributes(self):
        if self.verbose:
            print("[BiomartDataset:'%s'] Fetching attributes" % self.name)

        # retrieve default attributes from the dataset configuration page
        r = self.server.get_request(type="configuration", dataset=self.name)
        xml = fromstring(r.text)

        for idx, attribute_page in enumerate(xml.findall('./AttributePage')):

            name = attribute_page.get('internalName')
            display_name = attribute_page.get('displayName')

            default_attributes = []

            for attribute in attribute_page.findall('./*/*/AttributeDescription[@default="true"]'):
                default_attributes.append(attribute.get('internalName'))

            self._attribute_pages[name] = BiomartAttributePage(
                name,
                display_name = display_name,
                default_attributes = default_attributes,
                is_default = (idx == 0)  # first attribute page fetched is considered default
            )

        # grab attribute details
        r = self.server.get_request(type="attributes", dataset=self.name)
        for line in r.iter_lines():
            line = line.decode('utf8')
            if line:
                line = line.split("\t")
                page = line[3]
                name = line[0]

                if page not in self._attribute_pages:
                    self._attribute_pages[page] = BiomartAttributePage(page)
                    if self.verbose:
                        print("[BiomartDataset:'%s'] Warning: attribute page '%s' is not specified in server's configuration" % (self.name, page))

                attribute = BiomartAttribute(name=name, display_name=line[1])
                self._attribute_pages[page].add(attribute)

    def search(self, params={}, header=0, count=False, formatter='TSV'):
        if not isinstance(params, dict):
            raise Exception("'params' argument must be a dict")

        if self.verbose:
            print("[BiomartDataset:'%s'] Searching using following params:" % self.name)
            pprint.pprint(params)

        # read filters and attributes from params
        filters = params.get('filters', {})
        attributes = params.get('attributes', [])

        # check filters
        for filter_name, filter_value in filters.items():
            dataset_filter = self.filters.get(filter_name, None)

            if not dataset_filter:
                if self.verbose:
                    self.show_filters()
                raise Exception("The filter '%s' does not exist." % filter_name)

            accepted_values = dataset_filter.accepted_values
            if len(accepted_values) > 0:
                incorrect_value = None

                if (isinstance(filter_value, list) or isinstance(filter_value, tuple)) and dataset_filter.filter_type == 'list':
                    incorrect_value = filter(lambda v: v not in accepted_values, filter_value)
                elif filter_value not in accepted_values:
                    incorrect_value = filter_value

                if incorrect_value:
                    error_msg = "the value(s) '%s' for filter '%s' cannot be used." % (incorrect_value, filter_name)
                    error_msg += " Use values from: [%s]" % ", ".join(map(str, accepted_values))
                    raise Exception(error_msg)

        # check attributes unless we're only counting
        if not count:

            # discover attributes and pages
            self.fetch_attributes()

            # no attributes given, use default attributes
            if not attributes and self._attribute_pages:
                # get default attribute page
                page = next(filter(lambda attr_page: attr_page.is_default, self._attribute_pages.values()))
                
                # get default attributes from page
                attributes = [a.name for a in page.attributes.values() if a.is_default]

                # there is no default attributes, get all attributes from page
                if not attributes:
                    attributes = [a.name for a in page.attributes.values()]

            # if no default attributes have been defined, raise an exception
            if not attributes:
                raise Exception("at least one attribute is required, none given")

            for attribute_name in attributes:
                found = False
                for page in self._attribute_pages.values():
                    if attribute_name in page.attributes.keys():
                        found = True
                        break
                if not found:
                    if self.verbose:
                        self.show_attributes()
                    raise Exception("The attribute '%s' does not exist." % attribute_name)

            # guess the attribute page and check if all attributes belong to it.
            guessed_page = None

            for tested_page in self._attribute_pages.values():
                if set(attributes).issubset(tested_page.attributes.keys()):
                    guessed_page = tested_page
                    break

            if guessed_page is None:
                # selected attributes must belong to the same attribute page.
                if self.verbose:
                    self.show_attributes()
                raise Exception("You must use attributes that belong to the same attribute page.")
        # filters and attributes looks ok, start building the XML query
        root = Element('Query')
        root.attrib.update({
            'virtualSchemaName': self.database.virtual_schema,
            'formatter': 'TSV',
            'header': str(header),
            'uniqueRows': '1',
            'datasetConfigVersion': '0.6',
            'count': count is True and '1' or ''
        })

        dataset = SubElement(root, "Dataset")
        dataset.attrib.update({
            'name': self.name,
            'interface': self.interface
        })

        # Add filters to the XML query
        for filter_name, filter_value in filters.items():
            dataset_filter = self.filters[filter_name]

            filter_elem = SubElement(dataset, "Filter")
            filter_elem.set('name', filter_name)

            if dataset_filter.filter_type in ['boolean', 'boolean_list']:
                if filter_value is True or filter_value.lower() in ('included', 'only'):
                    filter_elem.set('excluded', '0')
                elif filter_value is False or filter_value.lower() == 'excluded':
                    filter_elem.set('excluded', '1')
            else:
                if isinstance(filter_value, list) or isinstance(filter_value, tuple):
                    filter_value = ",".join(map(str, filter_value))
                filter_elem.set('value', str(filter_value))

        # Add attributes to the XML query, unless we're only counting
        if not count:
            for attribute_name in attributes:
                attribute_elem = SubElement(dataset, "Attribute")
                attribute_elem.set('name', str(attribute_name))

        if self.verbose:
            print("[BiomartDataset] search query:\n%s" % tostring(root))

        return self.server.get_request(query = tostring(root))

    def count(self, params = {}):
        r = self.search(params, count = True)
        return int(r.text.strip())
    
class BiomartFilter(object):
    def __init__(self, name, display_name, accepted_values, filter_type):
        self.name = name
        self.display_name = display_name
        self.filter_type = filter_type
        self.accepted_values = accepted_values.replace('[', '').replace(']', '')

        if self.accepted_values:
            self.accepted_values = self.accepted_values.split(",")

        if not self.accepted_values and self.filter_type == 'boolean_list':
            self.accepted_values = [True, False, 'excluded', 'included', 'only']

    def __repr__(self):
        return "'%s' (type: %s, values: [%s])" % (
            self.display_name,
            self.filter_type,
            ", ".join(map(str, self.accepted_values))
        )
    
class BiomartAttribute(object):
    def __init__(self, name, display_name, is_default = False):
        self.name = name
        self.display_name = display_name
        self.is_default = is_default

    def __repr__(self):
        return "'%s' (default: %s)" % (self.display_name, self.is_default)
    
class BiomartAttributePage(object):
    def __init__(self, name, display_name=None, attributes=None, default_attributes=None, is_default=False):
        self.name = name
        self.display_name = display_name or name
        self.attributes = attributes if attributes else {}
        self.default_attributes = default_attributes if default_attributes else []
        self.is_default = is_default

    def add(self, attribute):
        attribute.is_default = attribute.name in self.default_attributes
        self.attributes[attribute.name] = attribute

    def __repr__(self):
        return "'%s': (attributes: %s, defaults: %s)" % (self.display_name, self.attributes, repr(self.default_attributes))
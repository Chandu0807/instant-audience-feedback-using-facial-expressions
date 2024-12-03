import pkg_resources
packages = pkg_resources.working_set
packages_list = ["%s==%s" % (i.key, i.version) for i in packages]
print(packages_list)
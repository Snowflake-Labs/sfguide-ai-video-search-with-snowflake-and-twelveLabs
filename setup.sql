use role ACCOUNTADMIN;

create database if not exists DASH_DB;
create schema if not exists DASH_SCHEMA;
create warehouse if not exists DASH_WH_S WAREHOUSE_SIZE=SMALL;

use database DASH_DB;
use schema DASH_SCHEMA;
use warehouse DASH_WH_S;

create stage if not exists DASH_UDFS;
create stage if not exists DASH_PKGS;

create compute pool if not exists CPU_X64_XS
  MIN_NODES = 1
  MAX_NODES = 5
  INSTANCE_FAMILY = CPU_X64_XS;

create role if not exists DASH_CONTAINER_RUNTIME_ROLE;
grant role DASH_CONTAINER_RUNTIME_ROLE to role ACCOUNTADMIN;
grant usage on database DASH_DB to role DASH_CONTAINER_RUNTIME_ROLE;
grant all on schema DASH_SCHEMA to role DASH_CONTAINER_RUNTIME_ROLE;
grant all on stage DASH_UDFS to role DASH_CONTAINER_RUNTIME_ROLE;
grant all on stage DASH_PKGS to role DASH_CONTAINER_RUNTIME_ROLE;
grant usage on warehouse DASH_WH_S to role DASH_CONTAINER_RUNTIME_ROLE;
grant all on compute pool CPU_X64_XS to role DASH_CONTAINER_RUNTIME_ROLE;

create network rule if not exists allow_all_rule
  TYPE = 'HOST_PORT'
  MODE= 'EGRESS'
  VALUE_LIST = ('0.0.0.0:443','0.0.0.0:80');

create external access integration if not exists allow_all_access_integration
  ALLOWED_NETWORK_RULES = (allow_all_rule)
  ENABLED = true;

grant usage on integration allow_all_access_integration to role DASH_CONTAINER_RUNTIME_ROLE;

-- TODO: Replace tlk_XXXXXXXXXXXXXXXXXX with your Twelve Labs API Key
create secret twelve_labs_api
  TYPE = GENERIC_STRING
  SECRET_STRING = 'tlk_XXXXXXXXXXXXXXXXXX';

grant all on secret twelve_labs_api TO ROLE DASH_CONTAINER_RUNTIME_ROLE;
  
create network rule twelvelabs_network_rule
  MODE = EGRESS
  TYPE = HOST_PORT
  VALUE_LIST = ('api.twelvelabs','api.twelvelabs.io','twelvelabs.io');

create external access integration twelvelabs_access_integration
  ALLOWED_NETWORK_RULES = (twelvelabs_network_rule)
  ALLOWED_AUTHENTICATION_SECRETS = (twelve_labs_api)
  ENABLED = true;

grant usage on integration twelvelabs_access_integration to role DASH_CONTAINER_RUNTIME_ROLE;

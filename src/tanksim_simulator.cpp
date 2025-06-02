#include "ros/forwards.h"
#include "ros/init.h"
#include "ros/publisher.h"
#include "ros/subscriber.h"
#include <cstdint>
#include <ode/collision.h>
#include <ode/collision_space.h>
#include <ode/common.h>
#include <ode/mass.h>
#include <ode/misc.h>
#include <ode/objects.h>
#include <ode/odecpp_collision.h>
extern "C" {
	#include <raylib.h>
    #include "raymath.h"
    #include <rlgl.h>
}
#include <ros/ros.h>
#include <ros/package.h>
#include "tanksim/actuator.h"
#include "tanksim/sensor.h"
#include "tanksim/info.h"
#include "tanksim/link.h"
#include <ode/ode.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <assert.h>

int sim_speed = 60;

// RES MGR

enum res_meshes {
    RES_MESH_TERRAIN,
    RES_MESH_TANK,
    RES_MESH_TANK_TURRET,
    RES_MESH_BUILDING,
    RES_MESH_CUBE,
    RES_MESH_TREE,
    RES_MESH_QUAD,
    RES_MESH_LEN
};

enum res_textures {
    RES_TEX_GRASS,
    RES_TEX_TANK_RED,
    RES_TEX_TANK_GREEN,
    RES_TEX_BUILDING,
    RES_TEX_FENCE,
    RES_TEX_LEN
};

enum res_materials {
    RES_MAT_DEFAULT,
    RES_MAT_TANK_RED,
    RES_MAT_TANK_GREEN,
    RES_MAT_BUILDING,
    RES_MAT_FENCE,
    RES_MAT_LEN
};

enum res_shaders {
    RES_SHADER_DEFAULT,
    RES_SHADER_INSTANCING,
    RES_SHADER_LEN
};

typedef struct {
    // meshes
    Model models[RES_MESH_LEN];
    Mesh meshes[RES_MESH_LEN];
    Matrix* transforms[RES_MESH_LEN];
    size_t batch_size[RES_MESH_LEN];
    size_t batch_capacity[RES_MESH_LEN];

    // textures
    Texture2D textures[RES_TEX_LEN];

    // materials
    Material materials[RES_MAT_LEN];

    // shaders
    Shader shaders[RES_SHADER_LEN];
} res_t;

std::vector<Vector3> pointcloud = {};

typedef struct {
    float* v;
    size_t w,h;
} heightfield_t;

heightfield_t heightfield_load_from_image(const char* file){
    Image img = LoadImage(file);
    heightfield_t heightfield = {0};
    heightfield.w = img.width;
    heightfield.h = img.height;
    heightfield.v = (float*)calloc(img.width*img.height,sizeof(float));

    // Image.data stores pixels as unsigned char (RGBA)
    Color* pixels = (Color*)img.data;  // raylib's Color is 4 bytes (r,g,b,a)

    for(int y = 0; y < heightfield.h; y++) {
        for(int x = 0; x < heightfield.w; x++) {
            Color px = pixels[y * heightfield.w + x];
            Color px2 = pixels[x*heightfield.w+y];
            float brightness = (px.r + px.g + px.b) / (3.0f * 255.0f);
            float b2 = (px2.r+px2.g+px2.b)/(3.0f*255.0f);
            if(brightness > 0.55) brightness *= 2;
            heightfield.v[y * heightfield.w + x] = brightness * 30.0f;
        }
    }

    UnloadImage(img);
    return heightfield;
}

Mesh res_mesh_gen_heightfield(heightfield_t* in){
    size_t width = in->w;
    size_t height = in->h;
    float* heightfield = in->v;

    Mesh mesh = {0};
    mesh.triangleCount = 2*width*height;
    mesh.vertexCount = 3*mesh.triangleCount;
    mesh.vertices = (float*)calloc(3*mesh.vertexCount,sizeof(float));
    mesh.texcoords = (float*)calloc(2*mesh.vertexCount,sizeof(float));
    size_t vtxIndex = 0;
    size_t texIndex = 0;

    for (size_t y = 0; y < height-1; y++) {
        for (size_t x = 0; x < width-1; x++) {
            // Heights at four corners
            float h00 = heightfield[y * width + x];
            float h10 = heightfield[y * width + (x + 1)];
            float h01 = heightfield[(y + 1) * width + x];
            float h11 = heightfield[(y + 1) * width + (x + 1)];

            // Positions of four corners (x,z are grid coords, y is height)
            Vector3 p00 = { (float)x-width/2,      h00, (float)y-height/2 };
            Vector3 p10 = { (float)(x+1)-width/2,  h10, (float)y-height/2 };
            Vector3 p01 = { (float)x-width/2,      h01, (float)(y+1)-height/2 };
            Vector3 p11 = { (float)(x+1)-width/2,  h11, (float)(y+1)-height/2 };

            // First triangle: p00, p10, p11
            mesh.vertices[vtxIndex++] = p00.x; mesh.vertices[vtxIndex++] = p00.y; mesh.vertices[vtxIndex++] = p00.z;
            mesh.vertices[vtxIndex++] = p10.x; mesh.vertices[vtxIndex++] = p10.y; mesh.vertices[vtxIndex++] = p10.z;
            mesh.vertices[vtxIndex++] = p11.x; mesh.vertices[vtxIndex++] = p11.y; mesh.vertices[vtxIndex++] = p11.z;

            mesh.texcoords[texIndex++] = 0; mesh.texcoords[texIndex++] = 0;
            mesh.texcoords[texIndex++] = 1; mesh.texcoords[texIndex++] = 0;
            mesh.texcoords[texIndex++] = 1; mesh.texcoords[texIndex++] = 1;

            // Second triangle: p00, p11, p01
            mesh.vertices[vtxIndex++] = p00.x; mesh.vertices[vtxIndex++] = p00.y; mesh.vertices[vtxIndex++] = p00.z;
            mesh.vertices[vtxIndex++] = p11.x; mesh.vertices[vtxIndex++] = p11.y; mesh.vertices[vtxIndex++] = p11.z;
            mesh.vertices[vtxIndex++] = p01.x; mesh.vertices[vtxIndex++] = p01.y; mesh.vertices[vtxIndex++] = p01.z;

            mesh.texcoords[texIndex++] = 0; mesh.texcoords[texIndex++] = 0;
            mesh.texcoords[texIndex++] = 1; mesh.texcoords[texIndex++] = 1;
            mesh.texcoords[texIndex++] = 0; mesh.texcoords[texIndex++] = 1;
        }
    }
    UploadMesh(&mesh, 0);
    return mesh;
}

void res_mesh_batch_push(res_t* res, size_t mesh_idx, Matrix transform){
    //printf("push mesh!\n");
    if(res->transforms[mesh_idx] == NULL || res->batch_size[mesh_idx] >= res->batch_capacity[mesh_idx]){
        //printf("abtch realloc!!!\n");
        res->batch_capacity[mesh_idx] = res->batch_capacity[mesh_idx]>0?res->batch_capacity[mesh_idx]*2:8;
        res->transforms[mesh_idx] = (Matrix*)realloc(res->transforms[mesh_idx],res->batch_capacity[mesh_idx]*sizeof(Matrix));
        memset(res->transforms[mesh_idx]+res->batch_size[mesh_idx],0,(res->batch_capacity[mesh_idx]-res->batch_size[mesh_idx])*sizeof(Matrix));
    }
    //printf("segfault? ");
    Matrix* transforms = res->transforms[mesh_idx];
    transforms[res->batch_size[mesh_idx]++] = transform;
    //printf("nah\n");
}

std::string asset(const std::string& filename) {
    static const std::string base_path = ros::package::getPath("tanksim");
    std::cout<<base_path<<"/"<<filename<<'\n';
    return base_path + "/" + filename;
}

res_t res_init(){

    res_t out = (res_t){0};
    out.models[RES_MESH_TANK] = LoadModel(asset("assets/t90c.obj").c_str());
    out.meshes[RES_MESH_TANK] = out.models[RES_MESH_TANK].meshes[0];
    out.meshes[RES_MESH_TANK_TURRET] = out.models[RES_MESH_TANK].meshes[1];
    out.meshes[RES_MESH_CUBE] = GenMeshCube(4,4,4);

    out.textures[RES_TEX_TANK_GREEN] = LoadTexture(asset("assets/t90_1.jpg").c_str());
    out.textures[RES_TEX_TANK_RED] = LoadTexture(asset("assets/t90_2.jpg").c_str());
    out.textures[RES_TEX_GRASS] = LoadTexture(asset("assets/tex_grass.jpg").c_str());
    GenTextureMipmaps(&out.textures[RES_TEX_GRASS]);
    SetTextureFilter(out.textures[RES_TEX_GRASS], TEXTURE_FILTER_TRILINEAR);

    out.shaders[RES_SHADER_DEFAULT] = LoadShader(0,0);
    out.shaders[RES_SHADER_INSTANCING] = LoadShader(asset("shaders/instancing.vs.glsl").c_str(), 0);

    for(size_t i = 0; i < RES_MAT_LEN; i++){
        out.materials[i] = LoadMaterialDefault();
    }

    out.materials[RES_MAT_DEFAULT].maps[MATERIAL_MAP_ALBEDO].texture = out.textures[RES_TEX_GRASS];

    out.materials[RES_MAT_TANK_GREEN].maps[MATERIAL_MAP_ALBEDO].texture = out.textures[RES_TEX_TANK_GREEN];
    out.materials[RES_MAT_TANK_GREEN].shader = out.shaders[RES_SHADER_INSTANCING];
    return out;
}

// QUADTREE

typedef struct {
    size_t id;
    float x,z;
} qt_key;

#define QT_MAX_KEYS 32
typedef struct qt_node {
    size_t children[4];
    size_t key_count;
    qt_key key[QT_MAX_KEYS];
    float x,z,w;
    uint8_t isleaf;
} qt_node;

typedef struct {
    qt_node* pool;
    size_t* freelist;
    size_t allocator_size, allocator_capacity;
    size_t freelist_size, freelist_capacity;
    size_t root;
} qt_tree;

size_t qt_node_alloc(qt_tree* tree){
    if(tree->freelist_size > 0){
        size_t out = tree->freelist[--tree->freelist_size];
        tree->pool[out] = (qt_node){0};
        tree->pool[out].isleaf = 1;
        assert(out<tree->allocator_size);
        return out;
    }

    if(tree->pool == NULL || tree->allocator_size >= tree->allocator_capacity){
        tree->allocator_capacity *= 2;
        //printf("new size: %zu bytes\n",tree->allocator_capacity*sizeof(qt_node));
        tree->pool = (qt_node*)realloc(tree->pool,tree->allocator_capacity*sizeof(qt_node));
        memset(tree->pool+tree->allocator_size,0,(tree->allocator_capacity-tree->allocator_size)*sizeof(qt_node));
    }
    tree->pool[tree->allocator_size] = (qt_node){0};
    tree->pool[tree->allocator_size].isleaf = 1;
    assert(tree->allocator_size<tree->allocator_capacity);
    return tree->allocator_size++;
}

void qt_node_free(qt_tree* tree, size_t node){
    if(tree->freelist == NULL || tree->freelist_size >= tree->freelist_capacity){
        tree->freelist_capacity *= 2;
        tree->freelist = (size_t*)realloc(tree->freelist,tree->freelist_capacity*sizeof(size_t));
        memset(tree->freelist+tree->freelist_size,0,(tree->freelist_capacity-tree->freelist_size)*sizeof(size_t));
    }
    tree->freelist[tree->freelist_size++] = node;
}

qt_tree qt_tree_init(float x, float z, float w){
    qt_tree out = {0};
    out.allocator_capacity = 16;
    out.freelist_capacity = 16;
    out.root = qt_node_alloc(&out);
    qt_node* root = &out.pool[out.root];
    root->x = x;
    root->z = z;
    root->w = w;
    return out;
}

void qt_tree_destroy(qt_tree* tree){
    if(tree == NULL) return;
    if(tree->pool != NULL) free(tree->pool);
    if(tree->freelist != NULL) free(tree->freelist);
}

size_t qt_node_child(qt_node* node, float x, float z){
    int right = x >= node->x;
    int bottom = z >= node->z;
    return (bottom << 1) | right; // quadrant index: 0 to 3
    /*if(x <= node->x && z <= node->z) return 0;
    else if(x > node->x && z <= node->z) return 1;
    else if(x <= node->x && z > node->z) return 2;
    else return 3;*/
}

void qt_node_insert(qt_node* node, qt_key key){
    assert(node->key_count < QT_MAX_KEYS);
    if(node->key_count >= QT_MAX_KEYS) return;
    node->key[node->key_count++] = key;
}

void qt_node_delete(qt_node* node, qt_key key){
    size_t idx;
    for(idx = 0; idx < node->key_count; idx++){
        if(node->key[idx].id == key.id) break;
    }
    if(idx == node->key_count) return; // key not found
    node->key_count--;
    memmove(&node->key[idx],&node->key[idx+1],(node->key_count-idx)*sizeof(qt_key));
}

void qt_node_split(qt_tree* tree, size_t node_idx, size_t depth){
    //printf("split\n");

    qt_node* node = &tree->pool[node_idx];
    float hw = node->w / 2.0f;
    for(size_t i = 0; i < 4; i++){
        size_t c = qt_node_alloc(tree);
        node = &tree->pool[node_idx];
        node->children[i] = c;
        qt_node* child = &tree->pool[node->children[i]];
        /*child->x = node->x-node->w/4+(node->w/2)*(i%2);
        child->z = node->z-node->w/4+(node->w/2)*(i/2);
        child->w = node->w/2;*/
        float dx = (i % 2 == 0) ? -0.5f : 0.5f;
        float dy = (i / 2 == 0) ? -0.5f : 0.5f;
        child->x = node->x + dx * hw;
        child->z = node->z + dy * hw;
        child->w = hw;
        child->isleaf = 1;
        //printf("new leaf node at depth %zu: x:%f z:%f w:%f\n",depth,child->x,child->z,child->w);
    }

    for(size_t i = 0; i < node->key_count; i++){
        size_t child_idx = qt_node_child(node,node->key[i].x,node->key[i].z);
        qt_node_insert(&tree->pool[node->children[child_idx]],node->key[i]);
    }
    memset(node->key,0,QT_MAX_KEYS*sizeof(qt_key));
    node->key_count = 0;
    node->isleaf = 0;
}

void qt_tree_insert(qt_tree* tree, size_t id, float x, float z){
    size_t node_idx = tree->root;
    qt_node* node = &tree->pool[node_idx];
    size_t depth = 0;
    while(node->isleaf == 0){
        //printf("testing %zu children\n",node_idx);
        node_idx = node->children[qt_node_child(node,x,z)];
        //printf("testing %zu the child\n",qt_node_child(node,x,z));
        node = &tree->pool[node_idx];
        depth++;
    }
    size_t d = 1;
    while(node->key_count >= QT_MAX_KEYS) {
        qt_node_split(tree,node_idx,depth+d);
        node = &tree->pool[node_idx];
        node_idx = node->children[qt_node_child(node, x, z)];
        node = &tree->pool[node_idx];
        d++;
    }
    //printf("insert (x:%f z:%f) to node %zu\n",x,z,node_idx);
    qt_node_insert(node,(qt_key){id,x,z});
}

void qt_node_merge(qt_tree* tree, qt_node* parent){
    //printf("merge!\n");
    parent->isleaf = 1;
    for(size_t i = 0; i < 4; i++){
        qt_node* child = &tree->pool[parent->children[i]];
        for(size_t j = 0; j < child->key_count; j++) qt_node_insert(parent,child->key[j]);
        qt_node_free(tree,parent->children[i]);
    }
    memset(parent->children,0,4*sizeof(size_t));
}

void qt_tree_delete(qt_tree* tree, qt_key key){
    qt_node* node = &tree->pool[tree->root];
    qt_node* parent = NULL;
    while(node->isleaf == 0){
        parent = node;
        node = &tree->pool[node->children[qt_node_child(node,key.x,key.z)]];
    }
    qt_node_delete(node,key);
    if(parent != NULL){
        size_t total_keys = 0;
        for(size_t i = 0; i < 4; i++) total_keys += tree->pool[parent->children[i]].key_count;
        if(total_keys < QT_MAX_KEYS){
            qt_node_merge(tree,parent);
            //node = parent;
        }
    }
}

float util_distance(float x1, float y1, float x2, float y2){
    float x3 = x1-x2;
    float y3 = y1-y2;
    return sqrt(x3*x3+y3*y3);
}

void qt_tree_find(qt_tree* tree, qt_node* node, float x, float z, float* nearest_dist, qt_key** nearest_key){
    if(node == NULL) return;
    float radius = (node->w/2.)*1.41421356237;
    float dtc = util_distance(x, z, node->x, node->z);

    if(dtc-radius > *nearest_dist) return;

    if(node->isleaf){
        for(int i = 0; i < node->key_count; i++){
            float dist = util_distance(x, z, node->key[i].x, node->key[i].z);
            if(*nearest_dist == -1 || dist < *nearest_dist){
                *nearest_dist = dist;
                *nearest_key = &node->key[i];
            }
        }
    }else{
        for(int i = 0; i < 4; i++) qt_tree_find(tree, &tree->pool[node->children[i]], x, z, nearest_dist, nearest_key);
    }
}

// OPEN DYNAMICS ENGINE
typedef struct {
    dWorldID world;
    dSpaceID space;
    dJointGroupID contact;
    dGeomID ray;
} ode_t;

ode_t ode_init(){
    dInitODE2(0);
    ode_t out = {0};
    out.world = dWorldCreate();
    out.space = dHashSpaceCreate(0);
    out.contact = dJointGroupCreate(0);
    out.ray = dCreateRay(0,10.);
    dGeomRaySetFirstContact(out.ray,1);
    dWorldSetGravity(out.world, 0, -9.81, 0);
    return out;
}

void ode_callback(void* data, dGeomID o1, dGeomID o2) {
    ode_t* ode = (ode_t*)data;
    const int MAX_CONTACTS = 8;
    dContact contact[MAX_CONTACTS];

    int numc = dCollide(o1, o2, MAX_CONTACTS, &contact[0].geom, sizeof(dContact));
    for (int i = 0; i < numc; i++) {
        if (dCalcVectorLength3(contact[i].geom.normal) < 1e-6) {
            ROS_WARN("Skipping degenerate contact normal!");
            continue;
        }
        contact[i].surface.mode = dContactBounce | dContactApprox1 | dContactSoftERP;
        contact[i].surface.mu = 2.5;
        contact[i].surface.bounce = 0.1;
        contact[i].surface.bounce_vel = 0.4;
        contact[i].surface.soft_cfm = 0.01;
        contact[i].surface.soft_erp = 0.2;

        dJointID c = dJointCreateContact(ode->world, ode->contact, &contact[i]);
        dJointAttach(c, dGeomGetBody(contact[i].geom.g1), dGeomGetBody(contact[i].geom.g2));
    }
}

void ode_update(ode_t* ode){
    dSpaceCollide(ode->space, ode, &ode_callback);
    dWorldQuickStep(ode->world, 1./60.);
    dJointGroupEmpty(ode->contact);
}


// ECS
#define ALIGN_UP(x, a) (((x) + (a) - 1) & ~((a) - 1))

enum ecs_components {
    ECS_POSITION,
    ECS_PHYSICS,
    ECS_VEHICLE,
    ECS_TURRET,
    ECS_HEALTH,
    ECS_AGENT,
    ECS_RENDER,
    ECS_STICKER,
    ECS_ACTUATOR,
    ECS_SENSOR,

    ECS_LEN
};

struct ecs_t;
typedef struct {
    void (*init)(struct ecs_t*, uint8_t*, size_t, size_t);
    void (*destroy)(struct ecs_t*, size_t, size_t);
    void (*update)(struct ecs_t*, size_t, size_t, size_t);
    size_t* sparse;
    uint8_t* dense;
    size_t* map;
    size_t size, capacity; // dense array
    size_t sparse_size, sparse_capacity; // sparse array
    size_t component_size;
} ecs_component;

typedef struct ecs_t {
    size_t size,freelist_size,freelist_capacity;
    size_t* freelist;
    ecs_component components[ECS_LEN];
} ecs_t;

ecs_component ecs_component_init(size_t size, void (*init)(ecs_t*,uint8_t*,size_t,size_t),void (*update)(ecs_t*,size_t,size_t,size_t),void (*destroy)(ecs_t*,size_t,size_t)){
    ecs_component out = {0};
    out.component_size = (size+alignof(max_align_t)-1) & ~(alignof(max_align_t)-1);
    out.capacity = 16;
    out.sparse_capacity = 16;
    out.init = init;
    out.update = update;
    out.destroy = destroy;
    return out;
}

uint8_t* ecs_component_add(ecs_t* ecs, uint8_t* arg, size_t type, size_t id){
    ecs_component* component = &ecs->components[type];

    size_t last = component->sparse_size;
    if(component->sparse_size <= id) component->sparse_size = id+1;
    while(component->sparse == NULL || component->sparse_size >= component->sparse_capacity){
        component->sparse_capacity *= 2;
        component->sparse = (size_t*)realloc(component->sparse,component->sparse_capacity*sizeof(size_t));
        for(size_t i = last; i < component->sparse_capacity; i++) component->sparse[i] = SIZE_MAX;
        last = component->sparse_capacity;
    }
    if(component->sparse[id] != SIZE_MAX) return component->dense+component->sparse[id]*component->component_size;
    component->sparse[id] = component->size;

    //if(arg != NULL){
        if(component->dense == NULL || component->size >= component->capacity){
            component->capacity *= 2;
            component->dense = (uint8_t*)realloc(component->dense,component->capacity*component->component_size);
            memset(component->dense+component->size*component->component_size,0,(component->capacity-component->size)*component->component_size);
            component->map = (size_t*)realloc(component->map,component->capacity*sizeof(size_t));
            for(size_t i = component->size; i < component->capacity; i++) component->map[i] = SIZE_MAX;
        }

        component->map[component->size] = id;
        if(component->init != NULL) component->init(ecs,arg,type,id);
    //}
    component->size++;
    return component->dense+component->sparse[id]*component->component_size;
}

void ecs_component_remove(ecs_t* ecs, size_t type, size_t id){
    ecs_component* component = &ecs->components[type];
    size_t dense_idx = component->sparse[id];
    size_t last_idx = component->size-1;
    if(dense_idx == SIZE_MAX) return;
    if(component->destroy != NULL) component->destroy(ecs,type,id);

    // swap pop
    if(dense_idx != last_idx){
        memcpy(component->dense+dense_idx*component->component_size, component->dense+last_idx*component->component_size, component->component_size);

        size_t moved_id = component->map[last_idx];
        component->map[dense_idx] = moved_id;
        component->sparse[moved_id] = dense_idx;
    }

    memset(component->dense+last_idx*component->component_size, 0, component->component_size);
    component->sparse[id] = SIZE_MAX;
    component->size--;
}

uint8_t* ecs_component_get_sparse(ecs_t* ecs, size_t type, size_t id){
    ecs_component* component = &ecs->components[type];
    if(component->sparse_size < id) return NULL;
    size_t dense_idx = component->sparse[id];
    if(dense_idx == SIZE_MAX) return NULL;
    return component->dense+dense_idx*component->component_size;
}

uint8_t* ecs_component_get_dense(ecs_t* ecs, size_t type, size_t didx){
    ecs_component* component = &ecs->components[type];
    return &component->dense[didx*component->component_size];
}

size_t ecs_alloc_id(ecs_t* ecs){
    if(ecs->freelist_size > 0) return ecs->freelist[--ecs->freelist_size];
    return ecs->size++;
}

void ecs_free_id(ecs_t* ecs, size_t id){
    if(ecs->freelist == NULL || ecs->freelist_size >= ecs->freelist_capacity){
        ecs->freelist_capacity *= 2;
        ecs->freelist = (size_t*)realloc(ecs->freelist,ecs->freelist_capacity*sizeof(size_t));
        memset(ecs->freelist+ecs->freelist_size,0,(ecs->freelist_capacity-ecs->freelist_size)*sizeof(size_t));
    }
    ecs->freelist[ecs->freelist_size++] = id;

    for(size_t i = 0; i < ECS_LEN; i++){
        if(ecs->components[i].sparse[id] != SIZE_MAX) ecs_component_remove(ecs,i,id);
    }
}


// quadtree position thing

// ACTUATOR COMPONENT
typedef struct {
    char* prefix;
    char* label;
    int count;
} ecs_actuator_arg;

typedef struct {
    ros::NodeHandle* nh;
    ros::Subscriber listener;
    float value[4];
} ecs_actuator_t;

void ecs_actuator_init(ecs_t* ecs, uint8_t* arg, size_t type, size_t id){
    ecs_component* component = &ecs->components[type];
    uint8_t* mem = component->dense+component->sparse[id]*component->component_size;

    ecs_actuator_arg* in = (ecs_actuator_arg*)arg;
    ecs_actuator_t* out = (ecs_actuator_t*)mem;
    memset(out->value,0,sizeof(float[4]));

    //ros::NodeHandle nh_scoped(in->prefix);
    out->nh = new ros::NodeHandle(in->prefix);
    printf("%s/%s\n",in->prefix,in->label);

    // evil lambda pointer capture (stealing!)
    out->listener = out->nh->subscribe<tanksim::actuator>(
        in->label, 10,
        [out](const tanksim::actuator::ConstPtr& msg) {
            size_t n = std::min((int)msg->actuator.size(), 4);
            for(size_t i = 0; i < n; i++){
                float v = msg->actuator[i];
                if (std::isnan(v) || std::isinf(v)) {
                    printf("Received invalid actuator[%zu] = %f, zeroing",i,v);
                    out->value[i] = 0.0f;
                } else {
                    out->value[i] = std::clamp(v, -1.0f, 1.0f);
                }
            }
        }
    );
}

void ecs_actuator_update(ecs_t* ecs, size_t type, size_t id, size_t didx){

}

void ecs_actuator_destroy(ecs_t* ecs, size_t type, size_t id){
    ecs_actuator_t* obj = (ecs_actuator_t*)ecs_component_get_sparse(ecs,type,id);
    if(obj == NULL) return;
    obj->listener.shutdown();
    delete obj->nh;
}

// POSITION COMPONENT
typedef struct {
    qt_tree* tree;
    float pos[3];
    uint8_t track;
} ecs_position_arg;

typedef struct {
    qt_tree* tree;
    float pos[16];
    float pos_last[16];
    float pos_delta[3];
    float pos_delta_last[3];
    float pos_tree[2];
    uint8_t track;
} ecs_position_t;

void ecs_position_init(ecs_t* ecs, uint8_t* arg, size_t type, size_t id){
    //printf("position init\n");
    ecs_component* component = &ecs->components[type];
    uint8_t* mem = component->dense+component->sparse[id]*component->component_size;

    ecs_position_arg* in = (ecs_position_arg*)arg;

    if(in->track == 1) qt_tree_insert(in->tree, id, in->pos[0], in->pos[2]);

    Matrix m = MatrixIdentity();
    m.m12 = in->pos[0];
    m.m13 = in->pos[1];
    m.m14 = in->pos[2];

    ecs_position_t out = {
        .tree = in->tree,
        .pos = {0},
        .pos_last = {0},
        .pos_delta = {0,0,0},
        .pos_delta_last = {0,0,0},
        .pos_tree = {in->pos[0],in->pos[2]},
        .track = in->track
    };
    memcpy(&out.pos,&m,sizeof(float[16]));
    memcpy(&out.pos_last,&m,sizeof(float[16]));

    *(ecs_position_t*)mem = out;
}

void ecs_position_update(ecs_t* ecs, size_t type, size_t id, size_t didx){
    ecs_component* component = &ecs->components[type];
    uint8_t* mem = component->dense+component->sparse[id]*component->component_size;
    ecs_position_t* obj = (ecs_position_t*)mem;

    memcpy(obj->pos_delta_last,obj->pos_delta,sizeof(float[3]));
    obj->pos_delta[0] = obj->pos[3]-obj->pos_last[3];
    obj->pos_delta[1] = obj->pos[7]-obj->pos_last[7];
    obj->pos_delta[2] = obj->pos[11]-obj->pos_last[11];
    if(obj->track == 1){
        float delta[2] = {0};
        obj->pos_last[3] = obj->pos[3];
        obj->pos_last[11] = obj->pos[11];

        // fix quadtree
        if(obj->pos_delta[0]+obj->pos_delta[2] != 0.){
        //if(obj->pos[12]-obj->pos_tree[0] + obj->pos[14]-obj->pos_tree[1] > 1){
            qt_tree_delete(obj->tree,(qt_key){id,obj->pos_tree[0],obj->pos_tree[1]});
            qt_tree_insert(obj->tree,id,obj->pos[3],obj->pos[11]);
            obj->pos_tree[0] = obj->pos[3];
            obj->pos_tree[1] = obj->pos[11];
        }
    }
}

void ecs_position_destroy(ecs_t* ecs, size_t type, size_t id){
    ecs_component* component = &ecs->components[type];
    uint8_t* mem = component->dense+component->sparse[id]*component->component_size;
    ecs_position_t* obj = (ecs_position_t*)mem;

    if(obj->track == 1) qt_tree_delete(obj->tree,(qt_key){id,obj->pos_tree[0],obj->pos_tree[1]});
}

// RENDER COMPONENT
typedef struct {
    res_t* res;
    size_t mesh;
} ecs_render_arg;

typedef struct {
    res_t* res;
    size_t mesh;
} ecs_render_t;

void ecs_render_init(ecs_t* ecs, uint8_t* arg, size_t type, size_t id){
    ecs_component* component = &ecs->components[type];
    uint8_t* mem = component->dense+component->sparse[id]*component->component_size;

    //ecs_render_arg* in = (ecs_render_arg*)arg;
    // mutate or whatever
    ecs_render_arg* in = (ecs_render_arg*)arg;
    ecs_render_t out = {in->res, in->mesh};
    *(ecs_render_t*)mem = out;
    //printf("new render comp mesh: %zu\n",out.mesh);
}

void ecs_render_update(ecs_t* ecs, size_t type, size_t id, size_t didx){
    // POSITION COMPONENT DEPENDENCY
    /*ecs_component* component = &ecs->components[ECS_POSITION];
    size_t dense_idx = component->sparse[id];
    if(dense_idx == SIZE_MAX){
        printf("COMPONENT RENDER REQUIRES POSITION, ABORTING\n");
        abort();
    }
    uint8_t* mem = component->dense+dense_idx*component->component_size;
    ecs_position_t* obj_pos = (ecs_position_t*)mem;*/

    ecs_position_t* obj_pos = (ecs_position_t*)ecs_component_get_sparse(ecs,ECS_POSITION,id);
    if(obj_pos == NULL) return; // object has no position component
    Matrix mat;
    memcpy(&mat,obj_pos->pos,sizeof(float[16]));

    // combined transforms for sticker entities
    size_t* parent = (size_t*)ecs_component_get_sparse(ecs,ECS_STICKER,id);
    if(parent != NULL){
        ecs_position_t* parent_pos = (ecs_position_t*)ecs_component_get_sparse(ecs,ECS_POSITION,*parent);
        if(parent_pos != NULL){
            Matrix mat2;
            memcpy(&mat2,parent_pos->pos,sizeof(float[16]));
            mat = MatrixMultiply(mat,mat2);
        }
    }

    ecs_render_t* obj = (ecs_render_t*)ecs_component_get_dense(ecs,type,didx);
    //printf("update render comp mesh: %zu\n",obj->mesh);
    res_mesh_batch_push(obj->res, obj->mesh, mat);
}

void ecs_render_destroy(ecs_t* ecs, size_t type, size_t id){
    return;
}

// STICKER COMPONENT -- inheritance blegh

void ecs_sticker_init(ecs_t* ecs, uint8_t* arg, size_t type, size_t id){
    ecs_component* component = &ecs->components[type];
    uint8_t* mem = component->dense+component->sparse[id]*component->component_size;

    //ecs_render_arg* in = (ecs_render_arg*)arg;
    // mutate or whatever
    size_t* in = (size_t*)arg;
    *(size_t*)mem = *in;
    //printf("new render comp mesh: %zu\n",out.mesh);
}

void ecs_sticker_update(ecs_t* ecs, size_t type, size_t id, size_t didx){
    size_t* parent = (size_t*)ecs_component_get_sparse(ecs,type,id);
    if(parent == NULL) return;
}

void ecs_sticker_destroy(ecs_t* ecs, size_t type, size_t id){
    size_t* parent = (size_t*)ecs_component_get_sparse(ecs,type,id);
    if(parent == NULL) return;
    //ecs_sticker_t* obj = (ecs_sticker_t*)ecs_component_get_sparse(ecs,type,id);
    //for(size_t i = 0; i < obj->count; i++) ecs_id_free(obj->children[i]);
}

// PHYSICS COMPONENT
typedef struct {
    ode_t* ode;
    float dim[3];
    uint8_t dynamic;
} ecs_physics_arg;

typedef struct {
    ode_t* ode;
    dGeomID geom;
    dBodyID body;
    dMass mass;
    uint8_t dynamic;
} ecs_physics_t;

void ecs_physics_init(ecs_t* ecs, uint8_t* arg, size_t type, size_t id){
    ecs_component* component = &ecs->components[type];
    uint8_t* mem = component->dense+component->sparse[id]*component->component_size;

    //ecs_render_arg* in = (ecs_render_arg*)arg;
    // mutate or whatever
    ecs_physics_arg* in = (ecs_physics_arg*)arg;

    // check position component
    ecs_position_t* pos = (ecs_position_t*)ecs_component_get_sparse(ecs,ECS_POSITION,id);
    if(pos == NULL) {
        printf("physics depends on position component!\n");
        abort();
    }
    // set ODE position to pos component position

    ecs_physics_t* out = (ecs_physics_t*)mem;
    out->ode = in->ode;

    out->geom = dCreateBox(out->ode->space, in->dim[0], in->dim[1], in->dim[2]);

    if(in->dynamic == 1){
        out->dynamic = 1;
        out->body = dBodyCreate(out->ode->world);
        dMassSetBox(&out->mass, 2, in->dim[0], in->dim[1], in->dim[2]);  // density, lx, ly, lz
        //dMassAdjust(&out->mass, 2.0);         // set total mass
        dBodySetMass(out->body, &out->mass);
        dBodySetPosition(out->body,(dReal)pos->pos[3],(dReal)pos->pos[7],(dReal)pos->pos[11]);
        dGeomSetBody(out->geom, out->body);
        //dBodySetLinearDamping(out->body, 0.05);
        dBodySetAngularDamping(out->body, 0.1);
    }else{
        dGeomSetOffsetPosition(out->geom, (dReal)pos->pos[3], (dReal)pos->pos[7], (dReal)pos->pos[11]);
    }

    // ecs_physics_t out = /{in->res, in->mesh};
    //*(ecs_physics_t*)mem = out;
    //printf("new render comp mesh: %zu\n",out.mesh);
}

void util_matrix_ode_float16(float* matrix, const dReal* pos, const dReal* rot){
    // row major float16 matrix update
    matrix[3] = pos[0];
    matrix[7] = pos[1];
    matrix[11] = pos[2];

    for(size_t i = 0; i < 9; i++) matrix[4*(i/3)+i%3] = rot[4*(i/3)+i%3];
}

void ecs_physics_update(ecs_t* ecs, size_t type, size_t id, size_t didx){
    ecs_position_t* pos = (ecs_position_t*)ecs_component_get_sparse(ecs,ECS_POSITION,id);
    if(pos == NULL) return; // object has no position component

    ecs_physics_t* obj = (ecs_physics_t*)ecs_component_get_dense(ecs,type,didx);
    
    const dReal* ode_position = dBodyGetPosition(obj->body);
    const dReal* ode_rotation = dBodyGetRotation(obj->body);
    util_matrix_ode_float16(pos->pos,ode_position,ode_rotation);
}

void ecs_physics_destroy(ecs_t* ecs, size_t type, size_t id){
    ecs_physics_t* obj = (ecs_physics_t*)ecs_component_get_sparse(ecs,type,id);
    dBodyDestroy(obj->body);
    dGeomDestroy(obj->geom);
}

//raycast

float util_raycast(float length, float step, float* origin, float* vec, heightfield_t* heightfield, qt_tree* quadtree){
    float pos[3] = {origin[0]+heightfield->w/2,origin[1],origin[2]+heightfield->w/2};
    for(int i = 0; i < (int)(length/step); i++){
        pos[0] += step*vec[0];
        pos[1] += step*vec[1];
        pos[2] += step*vec[2];

        /*if(pos[0] > 0 && pos[0] < heightfield->w && pos[2] > 0 && pos[2] < heightfield->h){
            if(heightfield->v[(int)(pos[2]*heightfield->w)+(int)pos[0]] >= pos[1]) return i*step;
        }*/
        int x = (int)pos[0];
        int z = (int)pos[2];
        if(x >= 0 && x < heightfield->w && z >= 0 && z < heightfield->h){
            int idx = z * heightfield->w + x;
            if(heightfield->v[idx] >= pos[1]) return i*step;
        }else return i*step;

        qt_key* nearest = NULL;
        float dist = -1;
        qt_tree_find(quadtree, &quadtree->pool[quadtree->root], pos[0],pos[2], &dist, &nearest);
        if(dist > 0 && dist < 2) return i*step;
        //step *= 1.1;
    }
    return -1;
}  

Vector3 util_gyro(Quaternion q_curr, Quaternion q_prev, float dt){
    Quaternion q_delta = QuaternionMultiply(q_curr, QuaternionInvert(q_prev));

    if (q_delta.w < 0) q_delta = QuaternionScale(q_delta, -1.0f);
    q_delta = QuaternionNormalize(q_delta);

    float angle = 2.0f*acosf(q_delta.w);
    float s = sqrtf(1.0f - q_delta.w * q_delta.w);

    Vector3 axis;
    if(s < 1e-6f) axis = (Vector3){1.0f, 0.0f, 0.0f}; // arbitrary axis
    else axis = (Vector3){q_delta.x/s,q_delta.y/s,q_delta.z/s};

    return Vector3Scale(axis,angle/dt); // rad/dt
}

// TURRET STRUCT (FORWARD DEC FOR SIMPLICITY)
typedef struct {
    size_t timer;
    size_t ammo;
    size_t hit_id;
    float turn[2];
    float fire;
} ecs_turret_t;

// SENSOR COMPONENT
enum ros_sensors {
    ROS_SENSOR_GPS,
    ROS_SENSOR_IMU,
    ROS_SENSOR_DIAG,
    ROS_SENSOR_LIDAR,
    ROS_SENSOR_CAMERA,
    ROS_SENSOR_TURRET
};

const size_t ros_sensor_payload[] =  {
    0,
    0,
    0,
    90*2,
    0
};

typedef struct {
    heightfield_t* heightfield;
    qt_tree* quadtree;
    char* prefix;
    char* label;
    size_t size;
    size_t type;
} ecs_sensor_arg;

typedef struct {
    ros::NodeHandle* nh;
    ros::Publisher publisher;
    heightfield_t* heightfield;
    qt_tree* quadtree;
    size_t size;
    size_t type;
    //uint8_t* mem;
} ecs_sensor_t;

void ecs_sensor_init(ecs_t* ecs, uint8_t* arg, size_t type, size_t id){
    ecs_component* component = &ecs->components[type];
    uint8_t* mem = component->dense+component->sparse[id]*component->component_size;

    ecs_sensor_arg* in = (ecs_sensor_arg*)arg;
    ecs_sensor_t* out = (ecs_sensor_t*)mem;
    out->size = in->size;
    out->type = in->type;
    out->heightfield = in->heightfield;
    out->quadtree = in->quadtree;
    //out->mem = (uint8_t*)calloc(out->size,sizeof(uint8_t));

    //ros::NodeHandle nh_scoped(in->prefix);
    out->nh = new ros::NodeHandle(in->prefix);
    printf("%s/%s\n",in->prefix,in->label);
    out->publisher = out->nh->advertise<tanksim::sensor>(in->label,10);
} 

void ode_callback_ray(void *data, dGeomID g1, dGeomID g2) {
    const int MAX_CONTACTS = 1;
    float* ray_hit = (float*)data;

    // Check collisions
    dContact contacts[MAX_CONTACTS];
    int c = dCollide(g1, g2, MAX_CONTACTS, &contacts[0].geom, sizeof(dContact));
    for(int i = 0; i < c; i++){
        if(ray_hit[3] == -1 || contacts[i].geom.depth < ray_hit[3]){
            ray_hit[0] = contacts[i].geom.pos[0];
            ray_hit[1] = contacts[i].geom.pos[1];
            ray_hit[2] = contacts[i].geom.pos[2];
            ray_hit[3] = contacts[i].geom.depth;
        }
    }
}

void ecs_sensor_update(ecs_t* ecs, size_t type, size_t id, size_t didx){
    ecs_sensor_t* obj = (ecs_sensor_t*)ecs_component_get_dense(ecs,ECS_SENSOR,didx);
    ecs_position_t* pos = (ecs_position_t*)ecs_component_get_sparse(ecs,ECS_POSITION,id);
    size_t* parent = (size_t*)ecs_component_get_sparse(ecs,ECS_STICKER,id);
    ecs_position_t* parent_pos = (ecs_position_t*)ecs_component_get_sparse(ecs,ECS_POSITION,*parent);
    if(parent_pos == NULL) printf("no parent pos\n\n");
    // allll le sensor
    // todo
    tanksim::sensor msg;
    msg.data.clear();
    //msg.header = obj->type;
    size_t offset = 0;

    switch(obj->type){
        case ROS_SENSOR_LIDAR:{
            ecs_physics_t* parent_phys = (ecs_physics_t*)ecs_component_get_sparse(ecs,ECS_PHYSICS,*parent);
            if(parent_phys == NULL) break;
            ode_t* ode = parent_phys->ode;
            
            Vector3 origin = (Vector3){parent_pos->pos[3],parent_pos->pos[7]+2,parent_pos->pos[11]};
            Matrix m_mount = *((Matrix*)parent_pos->pos);
            //Matrix m_mount;
            //memcpy(&m_mount,parent_pos->pos,sizeof(float[16]));
            //printf("%f %f %f \n",m_mount.m12,m_mount.m13,m_mount.m14);
            m_mount.m12 = 0; m_mount.m13 = 0; m_mount.m14 = 0;
            Quaternion q_mount = QuaternionFromMatrix(m_mount);
            Quaternion q_mount_inv = QuaternionInvert(q_mount);
            Vector3 e_mount = QuaternionToEuler(q_mount);

            //dGeomID ray = dCreateRay(0,50.);
            //dGeomRaySetClosestHit(ray,1);
            //dGeomRaySetFirstContact(ray,1);

            for(size_t i = 0; i < 90; i++){
                Quaternion q_yaw = QuaternionFromAxisAngle((Vector3){0,1,0},(i*4+((float)(rand()%40))*.1)*DEG2RAD);//((float)(rand()%3600))*0.1*DEG2RAD); // polar linear scan with noise
                Quaternion q_pitch = QuaternionFromAxisAngle((Vector3){1,0,0},/*10*DEG2RAD);*/(((float)(rand()%40))*0.1-2.)*DEG2RAD); // stochastic scan
                Quaternion q_delta = QuaternionMultiply(q_yaw, q_pitch);
                Quaternion q_sensor = QuaternionMultiply(q_mount,q_delta); 
                Vector3 v_forward = { 0, 0, 1 }; // forward in local space
                Vector3 v_direction = Vector3RotateByQuaternion(v_forward, q_sensor); // rotate by full sensor rotation
                Vector3 v_offset = Vector3Add(origin,Vector3Scale(v_direction,16));
                //Vector3 e_sensor = QuaternionToEuler(q_sensor); // sensor absolute angle
                float d = util_raycast(50,.5,(float*)(&origin),(float*)(&v_direction),obj->heightfield,obj->quadtree);
                //dRay ray = dCreateRay(dSpaceID space, dReal length)
                //double dv_direction[3] = {v_direction.x,v_direction.y,v_direction.z};
                //double dv_origin[3] = {origin.x,origin.x,origin.z};

                /*dGeomRaySet(ode->ray, origin.x,origin.y,origin.z, v_direction.x,v_direction.y,v_direction.z);

                float ray_hit[4] = {0}; // vec3 + depth
                ray_hit[3] = -1;
                dSpaceCollide2(ode->ray,(dGeomID)ode->space,ray_hit,&ode_callback_ray);
                float d = ray_hit[3];*/
                if(d == -1) continue;


                //if(i == 0){
                Vector3 v_end = Vector3Add(origin,Vector3Scale(v_direction,d));
                //Vector3 v_end2 = Vector3Add(origin,Vector3Scale(v_direction,100));
                //DrawLine3D(origin,v_end2,(Color){(i/90),(i/90),(i/90)});
                //DrawLine3D(origin,v_end,RED);
                //pointcloud.push_back(v_end);
                /*if(i == 0){
                    printf("\rlidar: dist %f, euler (deg): [%f %f %f]",d,e_sensor.x*RAD2DEG,e_sensor.y*RAD2DEG,e_sensor.z*RAD2DEG);
                }*/
                /*if(i == 30){
                    printf("\rsensor quat: %f %f %f %f, dvec: %f %f %f",q_sensor.x,q_sensor.y,q_sensor.z,q_sensor.w,v_direction.x,v_direction.y,v_direction.z);
                }*/
                msg.data.push_back(d);
                msg.data.push_back(q_delta.x);
                msg.data.push_back(q_delta.y);
                msg.data.push_back(q_delta.z);
                msg.data.push_back(q_delta.w);
            }
            //dGeomDestroy(ray);
        } break;
        case ROS_SENSOR_GPS: {
            // meter precision gps
            // simulated timing jitter
            if(rand()%6 == 0){
                msg.data.push_back(floor(parent_pos->pos[3]));
                msg.data.push_back(floor(parent_pos->pos[7]));
                msg.data.push_back(floor(parent_pos->pos[11]));
            }else return;
        } break;
        case ROS_SENSOR_IMU: {
            Matrix m_mount = *((Matrix*)parent_pos->pos);
            Quaternion q_mount = QuaternionFromMatrix(m_mount);
            msg.data.push_back(q_mount.x);
            msg.data.push_back(q_mount.y);
            msg.data.push_back(q_mount.z);
            msg.data.push_back(q_mount.w);

            Matrix m_last = *((Matrix*)parent_pos->pos_last);
            Quaternion q_last = QuaternionFromMatrix(m_last);

            Vector3 v_gyro = util_gyro(q_mount,q_last,sim_speed);
            msg.data.push_back(v_gyro.x);
            msg.data.push_back(v_gyro.y);
            msg.data.push_back(v_gyro.z);

            Vector3 v_accel = Vector3Scale(Vector3Subtract(*((Vector3*)parent_pos->pos_delta),*((Vector3*)parent_pos->pos_delta_last)),sim_speed);
            Vector3 v_accel_local = Vector3RotateByQuaternion(v_accel,QuaternionInvert(q_mount));
            msg.data.push_back(v_accel_local.x);
            msg.data.push_back(v_accel_local.y);
            msg.data.push_back(v_accel_local.z);
        } break;
        case ROS_SENSOR_CAMERA: {
            // 'computer vision'
            Matrix m_mount = *((Matrix*)parent_pos->pos);
            Quaternion q_mount = QuaternionFromMatrix(m_mount);
            ecs_component* component = &ecs->components[ECS_VEHICLE];
            for(size_t i = 0; i < component->size; i++){
                ecs_position_t* scan_pos = (ecs_position_t*)ecs_component_get_sparse(ecs,ECS_POSITION,component->map[i]);
                if(scan_pos == NULL || scan_pos == parent_pos) continue;
                Vector3 v_scan = (Vector3){scan_pos->pos[3],scan_pos->pos[7]+2,scan_pos->pos[11]};
                Vector3 v_origin = (Vector3){parent_pos->pos[3],parent_pos->pos[7]+2,parent_pos->pos[11]};
                Vector3 v_difference = Vector3Subtract(v_scan,v_origin);
                Vector3 v_direction = Vector3Normalize(v_difference);
                float ray_length = Vector3Length(v_direction);
                float dist = util_raycast(200,1,(float*)&v_origin,(float*)&v_direction,obj->heightfield,obj->quadtree);
                Vector3 v_hit = Vector3Add(v_origin,Vector3Scale(v_direction,dist));
                float hit_dist = Vector3Distance(v_scan,v_hit);
                if(hit_dist < 16){
                    msg.data.push_back(v_difference.x);
                    msg.data.push_back(v_difference.y);
                    msg.data.push_back(v_difference.z);
                }
            }
        } break;
        case ROS_SENSOR_DIAG: {
            float* hp = (float*)ecs_component_get_sparse(ecs,ECS_HEALTH,*parent);
            if(hp == NULL) break;
            msg.data.push_back(*hp);
        } break;
        case ROS_SENSOR_TURRET:{
            Quaternion q_rel = QuaternionFromMatrix(*((Matrix*)pos->pos));
            ecs_turret_t* turret = (ecs_turret_t*)ecs_component_get_sparse(ecs,ECS_TURRET,id);
            msg.data.push_back(q_rel.x);
            msg.data.push_back(q_rel.y);
            msg.data.push_back(q_rel.z);
            msg.data.push_back(q_rel.w);
            msg.data.push_back(turret->ammo);
            msg.data.push_back(turret->timer>0?0:1); // can fire or not fire
        } break;
    }
    obj->publisher.publish(msg);
}

void ecs_sensor_destroy(ecs_t* ecs, size_t type, size_t id){
    ecs_sensor_t* obj = (ecs_sensor_t*)ecs_component_get_sparse(ecs,type,id);
    if(obj == NULL) return;
    //if(obj->mem != NULL) free(obj->mem);
    obj->publisher.shutdown();
    delete obj->nh;
}

// HEALTH COMPONENT
void ecs_health_init(ecs_t* ecs, uint8_t* arg, size_t type, size_t id){
    ecs_component* component = &ecs->components[type];
    uint8_t* mem = component->dense+component->sparse[id]*component->component_size;
    float* in = (float*)arg;
    *(float*)mem = *(float*)arg;
}

void ecs_health_update(ecs_t* ecs, size_t type, size_t id, size_t didx){
    float* hp = (float*)ecs_component_get_dense(ecs,type,didx);
    if(*hp <= 0) ecs_free_id(ecs,id);
}

// TURRET COMPONENT

void ecs_turret_init(ecs_t* ecs, uint8_t* arg, size_t type, size_t id){
    ecs_component* component = &ecs->components[type];
    uint8_t* mem = component->dense+component->sparse[id]*component->component_size;
    ecs_turret_t* obj = (ecs_turret_t*)mem;
    memset(mem,0,sizeof(ecs_turret_t));
    obj->ammo = 128; // hardcoded ammo
}

void ecs_turret_update(ecs_t* ecs, size_t type, size_t id, size_t didx){
    ecs_turret_t* obj = (ecs_turret_t*)ecs_component_get_dense(ecs,type,didx);
    ecs_position_t* pos = (ecs_position_t*)ecs_component_get_sparse(ecs,ECS_POSITION,id);
    ecs_actuator_t* actuator = (ecs_actuator_t*)ecs_component_get_sparse(ecs,ECS_ACTUATOR,id);

    if(actuator != NULL){
        obj->turn[0] = std::clamp(actuator->value[0],-1.0f,1.0f);
        obj->turn[1] = std::clamp(actuator->value[1],-1.0f,1.0f);
        obj->fire = actuator->value[2];
    }

    const float turret_turn_spd = 0.05;

    Matrix m = *((Matrix*)pos->pos);
    Quaternion q_mount = QuaternionFromMatrix(m);
    Vector3 e_mount = QuaternionToEuler(q_mount);
    //e_mount.x = std::clamp(e_mount.x+0.05f,-.5f,.5f);
    //e_mount.x += obj->turn[1]*turret_turn_spd;
    e_mount.y += obj->turn[0]*turret_turn_spd;
    //printf("turret euler: %.1f %.1f\n",e_mount.x,e_mount.y);
    Quaternion q_delta = QuaternionFromEuler(e_mount.x,e_mount.y,e_mount.z);
    m = QuaternionToMatrix(q_delta);
    m.m12 = 0; m.m13 = 0; m.m14 = 0;
    memcpy(pos->pos,&m,sizeof(float[16])); 

    if(obj->timer > 0) obj->timer--;
    if(obj->fire > .8 && obj->timer == 0 && obj->ammo > 0){
        obj->timer = 300;
        if(obj->ammo > 0) obj->ammo--;

        // fire turret
        size_t* parent = (size_t*)ecs_component_get_sparse(ecs,ECS_STICKER,id);
        if(parent == NULL) return;
        ecs_position_t* parent_pos = (ecs_position_t*)ecs_component_get_sparse(ecs,ECS_POSITION,*parent);
        if(parent_pos == NULL) return;
        ecs_sensor_t* sensor = (ecs_sensor_t*)ecs_component_get_sparse(ecs,ECS_SENSOR,id);
        Matrix mp = *((Matrix*)parent_pos->pos);
        Quaternion q_parent = QuaternionFromMatrix(mp);
        Quaternion q_world = QuaternionMultiply(q_parent,q_delta);
        Vector3 v_origin = (Vector3){parent_pos->pos[3],parent_pos->pos[7]+1,parent_pos->pos[11]};
        Vector3 v_forward = (Vector3){0,0,1};
        Vector3 v_direction = Vector3RotateByQuaternion(v_forward,q_world);
        float d = util_raycast(200,1,(float*)&v_origin, (float*)&v_direction,sensor->heightfield,sensor->quadtree);
        if(d > 0){
            // hit something
            Vector3 v_hit = Vector3Add(v_origin,Vector3Scale(v_direction,d));

            qt_key* nearest = NULL;
            float dist = -1;
            qt_tree_find(sensor->quadtree, &sensor->quadtree->pool[sensor->quadtree->root], v_hit.x,v_hit.z, &dist, &nearest);
            size_t hit_id = nearest->id;
            ecs_position_t* hit_pos = (ecs_position_t*)ecs_component_get_sparse(ecs,ECS_POSITION,hit_id);
            if(hit_pos != NULL){
                Vector3 vd = (Vector3){hit_pos->pos[3]-v_hit.x,hit_pos->pos[7]-v_hit.y,hit_pos->pos[11]-v_hit.z};
                dist = sqrt(vd.x*vd.x + vd.y*vd.y + vd.z*vd.z);
                if(dist < 4){ // 4m blast radius
                    obj->hit_id = hit_id;
                    float* hit_hp = (float*)ecs_component_get_sparse(ecs,ECS_HEALTH,hit_id);
                    if(hit_hp != NULL) *hit_hp -= 4-dist;
                }
            }
        }
    }
}

void ecs_turret_destroy(ecs_t* ecs, size_t type, size_t id){
    return;
}

// TANK COMPONENT
typedef struct {
    ode_t* ode;
    size_t wheels;
    float dim[3];
} ecs_vehicle_arg;

typedef struct {
    ode_t* ode;
    size_t wheel_count;
    dBodyID bwheels[8];
    dGeomID gwheels[8];
    dJointID jwheels[8];
    dJointGroupID joints;
    dMass mass;
    float throttle[2]; // control interface
    float brake[2];
    float steer;
} ecs_vehicle_t;

void ecs_vehicle_init(ecs_t* ecs, uint8_t* arg, size_t type, size_t id){
    ecs_component* component = &ecs->components[type];
    uint8_t* mem = component->dense+component->sparse[id]*component->component_size;

    //ecs_render_arg* in = (ecs_render_arg*)arg;
    // mutate or whatever
    ecs_vehicle_arg* in = (ecs_vehicle_arg*)arg;

    // check position component
    ecs_position_t* obj_pos = (ecs_position_t*)ecs_component_get_sparse(ecs,ECS_POSITION,id);
    ecs_physics_t* obj_phys = (ecs_physics_t*)ecs_component_get_sparse(ecs,ECS_PHYSICS,id);
    if(obj_pos == NULL || obj_phys == NULL){
        printf("physics depends on position AND physics component!\n");
        abort();
    }
    // build a tank

    //const dReal* lp = dBodyGetPosition(obj_phys->body);
    float x,y,z;
    x = obj_pos->pos[3];
    y = obj_pos->pos[7];
    z = obj_pos->pos[11];;
    /*x = (float)lp[0];
    y = (float)lp[1];
    z = (float)lp[2];*/

    ecs_vehicle_t* out = (ecs_vehicle_t*)mem;
    out->ode = in->ode;
    out->wheel_count = in->wheels;
    dMassSetSphere(&out->mass, .5, .5);// (&out->mass,2,);
    //dMassAdjust(&out->mass,1);
    out->joints = dJointGroupCreate(in->wheels);

    for(size_t i = 0; i < in->wheels; i++){
        //out->gwheels[i] = dCreateCylinder(out->ode->space,.5,.5);
        out->gwheels[i] = dCreateSphere(out->ode->space,.5);
        out->bwheels[i] = dBodyCreate(out->ode->world);
        dBodySetMass(out->bwheels[i],&out->mass);
        //dBodySetFiniteRotationMode(out->bwheels[i], 1 );
        dBodySetAutoDisableFlag(out->bwheels[i], 0 );
        dBodySetLinearDamping(out->bwheels[i], 0.05);
        dBodySetAngularDamping(out->bwheels[i], 0.1);

        int side = (i < in->wheels / 2) ? -1 : 1;
        int rowIndex = i % (in->wheels / 2);

        float xo,yo,zo;
        //xo = (in->dim[0]/2+1)*((i/(in->wheels/2))*2-1);
        yo = -.25;
        //zo = (in->dim[2]/(in->wheels/4)*(i%(in->wheels/2)))-(in->dim[2]/2);
        xo = (in->dim[0] / 2 +.5) * side;
        zo = -in->dim[2] / 2 + rowIndex * (in->dim[2]/(in->wheels/2));

        dGeomSetBody(out->gwheels[i],out->bwheels[i]);
        dBodySetPosition(out->bwheels[i],x+xo,y+yo,z+zo);

        dJointID joint = dJointCreateHinge(out->ode->world, out->joints);
        dJointAttach(joint, out->bwheels[i], obj_phys->body);
        const dReal* lp = dBodyGetPosition(out->bwheels[i]);
        dJointSetHingeAnchor(joint, lp[0],lp[1],lp[2]);//x+xo,y+yo,z+zo);
        dJointSetHingeAxis(joint,1,0,0);
        /*dReal a1[3] = {0,0,1};
        dReal a2[3] = {0,1,0};
        dJointSetHinge2Axes(joint, a1,a2);
        */
        //dJointSetHinge2Param(joint, dParamLoStop, -0.5);
        //dJointSetHinge2Param(joint, dParamHiStop, 0.5);
        dJointSetHingeParam(joint, dParamVel, 0);
        dJointSetHingeParam(joint, dParamFMax, 0);

        dJointSetHingeParam(joint, dParamSuspensionERP, 0.7);
        dJointSetHingeParam(joint, dParamSuspensionCFM, 0.0025);

        out->jwheels[i] = joint;
    }
}

void ecs_vehicle_update(ecs_t* ecs, size_t type, size_t id, size_t didx){
    ecs_vehicle_t* obj = (ecs_vehicle_t*)ecs_component_get_dense(ecs,type,didx);

    // actuators
    ecs_actuator_t* actuator = (ecs_actuator_t*)ecs_component_get_sparse(ecs,ECS_ACTUATOR,id);
    if(actuator != NULL){
        obj->throttle[0] = std::clamp(actuator->value[0], -1.f,1.f);
        obj->throttle[1] = std::clamp(actuator->value[1], -1.f,1.f);
        obj->brake[0] = std::clamp(actuator->value[2], 0.f,1.f);
        obj->brake[1] = std::clamp(actuator->value[3], 0.f,1.f);
    }
    
    for(size_t i = 0; i < obj->wheel_count; i++){
        float torque = obj->throttle[i/(obj->wheel_count/2)]*100.;
        //dBodyAddTorque(obj->bwheels[i], torque*.5, 0, 0);
        //if(torque > 0){
            dJointSetHingeParam(obj->jwheels[i], dParamVel, 20*obj->throttle[i/(obj->wheel_count/2)]);
            dJointSetHingeParam(obj->jwheels[i], dParamFMax, fabs(torque));
        //}
        float brake = 1-(obj->brake[i/(obj->wheel_count/2)]*.1);
        const dReal* av = dBodyGetAngularVel(obj->bwheels[i]);
        dBodySetAngularVel(obj->bwheels[i], av[0]*brake,av[1]*brake,av[2]*brake);
        /*const dReal* lp = dBodyGetPosition(obj->bwheels[i]);
        DrawCube((Vector3){(float)lp[0],(float)lp[1],(float)lp[2]}, .5,.5,.5, RED);*/
    }
}

void ecs_vehicle_destroy(ecs_t* ecs, size_t type, size_t id){
    ecs_vehicle_t* obj = (ecs_vehicle_t*)ecs_component_get_sparse(ecs,type,id);
    for(size_t i = 0; i < obj->wheel_count; i++){
        dBodyDestroy(obj->bwheels[i]);
        dGeomDestroy(obj->gwheels[i]);
        dJointDestroy(obj->jwheels[i]); 
    }
}


ecs_t ecs_init(){
    ecs_t out = (ecs_t){0};

    out.components[ECS_POSITION] = ecs_component_init(sizeof(ecs_position_t),ecs_position_init,ecs_position_update,ecs_position_destroy);
    out.components[ECS_RENDER] = ecs_component_init(sizeof(ecs_render_t),ecs_render_init,ecs_render_update,ecs_render_destroy);
    out.components[ECS_PHYSICS] = ecs_component_init(sizeof(ecs_physics_t),ecs_physics_init,ecs_physics_update,ecs_physics_destroy);
    out.components[ECS_VEHICLE] = ecs_component_init(sizeof(ecs_vehicle_t),ecs_vehicle_init,ecs_vehicle_update,ecs_vehicle_destroy);
    out.components[ECS_STICKER] = ecs_component_init(sizeof(size_t),ecs_sticker_init,ecs_sticker_update,ecs_sticker_destroy);
    out.components[ECS_ACTUATOR] = ecs_component_init(sizeof(ecs_actuator_t),ecs_actuator_init,ecs_actuator_update,ecs_actuator_destroy);
    out.components[ECS_SENSOR] = ecs_component_init(sizeof(ecs_sensor_t),ecs_sensor_init,ecs_sensor_update,ecs_sensor_destroy);
    out.components[ECS_TURRET] = ecs_component_init(sizeof(ecs_turret_t),ecs_turret_init,ecs_turret_update,ecs_turret_destroy);
    out.components[ECS_HEALTH] = ecs_component_init(sizeof(float),ecs_health_init,ecs_health_update,NULL);
    return out;
}

void ecs_destroy(ecs_t* ecs){
    if(ecs->freelist != NULL) free(ecs->freelist);
    for(size_t i = 0; i < ECS_LEN; i++){
        ecs_component* component = &ecs->components[i];
        if(component->sparse != NULL) free(component->sparse);
        if(component->dense != NULL) free(component->dense);
        if(component->map != NULL) free(component->map);
    }
}

void util_camera(Camera* camera){
    camera->position.x += IsKeyDown(KEY_D)-IsKeyDown(KEY_A);
    camera->position.z += IsKeyDown(KEY_W)-IsKeyDown(KEY_S);
    camera->position.y += IsKeyDown(KEY_SPACE)-IsKeyDown(KEY_LEFT_SHIFT);

    camera->target.x += IsKeyDown(KEY_D)-IsKeyDown(KEY_A);
    camera->target.z += IsKeyDown(KEY_W)-IsKeyDown(KEY_S);
    camera->target.y += IsKeyDown(KEY_SPACE)-IsKeyDown(KEY_LEFT_SHIFT);
}

dHeightfieldDataID util_ode_heightfield(heightfield_t* heightfield){
    dHeightfieldDataID data = dGeomHeightfieldDataCreate();

    //dGeomHeightfieldDataBuildSingle(dHeightfieldDataID d, const float *pHeightData, int bCopyHeightData, dReal width, dReal depth, int widthSamples, int depthSamples, dReal scale, dReal offset, dReal thickness, int bWrap)
    dGeomHeightfieldDataBuildSingle(
        data,
        heightfield->v,     // pointer to your heightfield
        0,                    // copy data? 0 = use in-place
        heightfield->w, heightfield->h,        // total width and depth in world units
        heightfield->w, heightfield->h, // sample resolution (data size)
        1.,         // vertical scaling factor (Y multiplier)
        0.0f,                 // base height offset
        1,
        1                    // wrap? usually 0 (false)
    );

    return data;
}

std::vector<uint32_t> link_buffer = {};
std::vector<uint8_t> team_buffer = {};
std::mutex link_mutex;
heightfield_t* global_hf = NULL;
bool ros_link(tanksim::link::Request &req, tanksim::link::Response &res){
    std::lock_guard<std::mutex> lock(link_mutex);

    if (link_buffer.empty()) {
        ROS_WARN("No tanks left to claim!");
        res.id = UINT32_MAX;
        return true;
    }

    res.id = link_buffer.back(); link_buffer.pop_back();
    res.team = team_buffer.back(); team_buffer.pop_back();
    res.speed = sim_speed;
    res.width = global_hf->w;
    res.height = global_hf->h;
    ROS_INFO("Assigned %d", res.id);
    return true;
}

int main(int argc, char** argv){
    ros::init(argc,argv,"tanksim_simulator");
    ros::NodeHandle nh;
	// Initialize the window
    const int screenWidth = 800;
    const int screenHeight = 600;
    InitWindow(screenWidth, screenHeight, "Tank Engine");

    Mesh meshes[RES_MESH_LEN];


    // Set the target FPS
    SetTargetFPS(sim_speed);
    Camera3D camera = (Camera3D){(Vector3){10,10,10},(Vector3){0,0,0},(Vector3){0,1,0},80,CAMERA_PERSPECTIVE};
    //SetCameraMode(camera, CAMERA_ORBITAL);
    qt_tree quadtree = qt_tree_init(0,0,8192);

    ecs_t ecs = ecs_init();
    res_t res = res_init();
    ode_t ode = ode_init();

    size_t objs = 0;
    size_t camera_idx = 0;
    size_t camera_mode = 0;
    float camera_zoom = 0;

    heightfield_t heightfield = heightfield_load_from_image(asset("assets/img_heightfield.png").c_str());
    res.meshes[RES_MESH_TERRAIN] = res_mesh_gen_heightfield(&heightfield);

    global_hf = &heightfield;
    ros::ServiceServer server = nh.advertiseService("link", ros_link); // here

    dHeightfieldDataID heightfield_ode_data = util_ode_heightfield(&heightfield);
    dGeomID heightfield_ode = dCreateHeightfield(ode.space, heightfield_ode_data, 1);

    rlDisableBackfaceCulling();

    // combat game
    ros::Publisher mission = nh.advertise<tanksim::sensor>("mission",10);
    std::vector<Vector2> objectives = {};


    printf("OpenGL version: %d\n", rlGetVersion());
    int frame = 0;
    // Main game loop
    while (!WindowShouldClose()) {    // Detect window close button or ESC key
        frame++;
        util_camera(&camera);
        // Start drawing
        BeginDrawing();

        ClearBackground(GRAY);

        // Draw your stuff here
        DrawFPS(5, 5);
        DrawText(TextFormat("entities: %zu",objs), 5, 20, 10, BLACK);
        //DrawText("Hello, Raylib!", 190, 200, 20, LIGHTGRAY);
        if(IsKeyPressed(KEY_X)){ //qt_tree_insert(&quadtree,objs++,rand()%4096-2048,rand()%4096-2048);
            for(size_t i = 0; i < 1; i++){
                float x = ((float)(rand()%800-400));
                float z = ((float)(rand()%800-400));
                float hp = 20;
                size_t id = ecs_alloc_id(&ecs);
                ecs_position_arg arg = (ecs_position_arg){&quadtree,x,heightfield.v[(int)floor(x+heightfield.w/2)*heightfield.w+(int)floor(z+heightfield.h/2)],z,1};
                ecs_component_add(&ecs,(uint8_t*)(&arg),ECS_POSITION,id);
                ecs_render_arg arg_render = (ecs_render_arg){&res,RES_MESH_TANK};
                ecs_component_add(&ecs,(uint8_t*)(&arg_render),ECS_RENDER,id);
                ecs_physics_arg arg_physics = (ecs_physics_arg){&ode,{3,1,5},1};
                ecs_component_add(&ecs,(uint8_t*)(&arg_physics),ECS_PHYSICS,id);
                ecs_vehicle_arg arg_vehicle = (ecs_vehicle_arg){&ode,6,{3,1,5}};
                ecs_component_add(&ecs,(uint8_t*)(&arg_vehicle),ECS_VEHICLE,id);
                ecs_component_add(&ecs,(uint8_t*)&hp,ECS_HEALTH,id);
                char prefix[16];
                snprintf(prefix,16,"tank_%zu/",ecs.components[ECS_VEHICLE].size-1);
                char label[16];
                snprintf(label,16,"actuator/drive");

                ecs_actuator_arg arg_actuator = (ecs_actuator_arg){prefix,label,4};
                ecs_component_add(&ecs,(uint8_t*)(&arg_actuator),ECS_ACTUATOR,id);

                // turret
                size_t pid = id;
                id = ecs_alloc_id(&ecs);
                arg = (ecs_position_arg){&quadtree,0,0,0,0};
                ecs_component_add(&ecs,(uint8_t*)(&arg),ECS_POSITION,id);
                arg_render = (ecs_render_arg){&res,RES_MESH_TANK_TURRET};
                ecs_component_add(&ecs,(uint8_t*)(&arg_render),ECS_RENDER,id);
                ecs_component_add(&ecs,(uint8_t*)&pid,ECS_STICKER,id);
                snprintf(label,16,"actuator/twist");
                arg_actuator = (ecs_actuator_arg){prefix,label,2};
                ecs_component_add(&ecs,(uint8_t*)(&arg_actuator),ECS_ACTUATOR,id);
                snprintf(label,16,"sensor/turret");
                ecs_sensor_arg arg_sensor = (ecs_sensor_arg){&heightfield,&quadtree,prefix,label,ros_sensor_payload[ROS_SENSOR_TURRET],ROS_SENSOR_TURRET};
                ecs_component_add(&ecs,(uint8_t*)(&arg_sensor),ECS_SENSOR,id);
                ecs_component_add(&ecs,NULL,ECS_TURRET,id);

                // lidar
                id = ecs_alloc_id(&ecs);
                arg = (ecs_position_arg){&quadtree,0,0,0,0};
                ecs_component_add(&ecs,(uint8_t*)(&arg),ECS_POSITION,id);
                ecs_component_add(&ecs,(uint8_t*)&pid,ECS_STICKER,id);
                snprintf(label,16,"sensor/lidar");
                arg_sensor = (ecs_sensor_arg){&heightfield,&quadtree,prefix,label,ros_sensor_payload[ROS_SENSOR_LIDAR],ROS_SENSOR_LIDAR};
                ecs_component_add(&ecs,(uint8_t*)(&arg_sensor),ECS_SENSOR,id);

                // gps
                id = ecs_alloc_id(&ecs);
                arg = (ecs_position_arg){&quadtree,0,0,0,0};
                ecs_component_add(&ecs,(uint8_t*)(&arg),ECS_POSITION,id);
                ecs_component_add(&ecs,(uint8_t*)&pid,ECS_STICKER,id);
                snprintf(label,16,"sensor/gps");
                arg_sensor = (ecs_sensor_arg){&heightfield,&quadtree,prefix,label,ros_sensor_payload[ROS_SENSOR_GPS],ROS_SENSOR_GPS};
                ecs_component_add(&ecs,(uint8_t*)(&arg_sensor),ECS_SENSOR,id);

                // imu
                id = ecs_alloc_id(&ecs);
                arg = (ecs_position_arg){&quadtree,0,0,0,0};
                ecs_component_add(&ecs,(uint8_t*)(&arg),ECS_POSITION,id);
                ecs_component_add(&ecs,(uint8_t*)&pid,ECS_STICKER,id);
                snprintf(label,16,"sensor/imu");
                arg_sensor = (ecs_sensor_arg){&heightfield,&quadtree,prefix,label,ros_sensor_payload[ROS_SENSOR_IMU],ROS_SENSOR_IMU};
                ecs_component_add(&ecs,(uint8_t*)(&arg_sensor),ECS_SENSOR,id);

                // camera
                id = ecs_alloc_id(&ecs);
                arg = (ecs_position_arg){&quadtree,0,0,0,0};
                ecs_component_add(&ecs,(uint8_t*)(&arg),ECS_POSITION,id);
                ecs_component_add(&ecs,(uint8_t*)&pid,ECS_STICKER,id);
                snprintf(label,16,"sensor/camera");
                arg_sensor = (ecs_sensor_arg){&heightfield,&quadtree,prefix,label,ros_sensor_payload[ROS_SENSOR_CAMERA],ROS_SENSOR_CAMERA};
                ecs_component_add(&ecs,(uint8_t*)(&arg_sensor),ECS_SENSOR,id);

                // diag
                id = ecs_alloc_id(&ecs);
                arg = (ecs_position_arg){&quadtree,0,0,0,0};
                ecs_component_add(&ecs,(uint8_t*)(&arg),ECS_POSITION,id);
                ecs_component_add(&ecs,(uint8_t*)&pid,ECS_STICKER,id);
                snprintf(label,16,"sensor/diag");
                arg_sensor = (ecs_sensor_arg){&heightfield,&quadtree,prefix,label,ros_sensor_payload[ROS_SENSOR_DIAG],ROS_SENSOR_DIAG};
                ecs_component_add(&ecs,(uint8_t*)(&arg_sensor),ECS_SENSOR,id);

                link_buffer.push_back(objs);
                team_buffer.push_back((uint8_t)(rand()%2));
                objs++;
            }
        }

        if(IsKeyPressed(KEY_Z)) camera_mode = (camera_mode+1)%2;
        if(IsKeyDown(KEY_UP)) camera_zoom += 0.05;
        else if(IsKeyDown(KEY_DOWN)) camera_zoom -= 0.05;
        if(camera_zoom < 0) camera_zoom = 0;
        if(ecs.components[ECS_VEHICLE].size > 0){
            size_t cidx = camera_idx;
            if(IsKeyPressed(KEY_C)) camera_idx++;
            camera_idx = camera_idx%ecs.components[ECS_VEHICLE].size;
            size_t camera_sparse_idx = ecs.components[ECS_VEHICLE].map[camera_idx];
            ecs_position_t* camera_pos = (ecs_position_t*)ecs_component_get_sparse(&ecs,ECS_POSITION, camera_sparse_idx);
            if(camera_pos != NULL){

                if(camera_mode == 1){
                    Matrix camera_matrix = *(Matrix*)camera_pos->pos;
                    Quaternion camera_quaternion = QuaternionFromMatrix(camera_matrix);
                    Vector3 camera_offset = Vector3RotateByQuaternion((Vector3){0,0,1},camera_quaternion);

                    camera.position = (Vector3){
                        camera_pos->pos[3],
                        camera_pos->pos[7]+2,
                        camera_pos->pos[11]
                    };

                    camera.target = Vector3Add(camera.position,camera_offset);
                    UpdateCamera(&camera,CAMERA_FIRST_PERSON);
                }else{
                    camera.target = (Vector3){
                        camera_pos->pos[3],
                        camera_pos->pos[7]+2,
                        camera_pos->pos[11]
                    };

                    if(cidx != camera_idx){
                        // switch pressed
                        camera.position = Vector3Add(camera.target,(Vector3){4,4,4});
                    }

                    UpdateCamera(&camera,CAMERA_THIRD_PERSON);
                }
            }
        }


        ode_update(&ode);
        if(ros::ok()) ros::spinOnce();

        // todo make this not gross

        for(size_t i = 0; i < ecs.components[ECS_POSITION].size; i++) ecs.components[ECS_POSITION].update(&ecs,ECS_POSITION,ecs.components[ECS_POSITION].map[i],i);
        for(size_t i = 0; i < ecs.components[ECS_RENDER].size; i++) ecs.components[ECS_RENDER].update(&ecs,ECS_RENDER,ecs.components[ECS_RENDER].map[i],i);
        for(size_t i = 0; i < ecs.components[ECS_PHYSICS].size; i++) ecs.components[ECS_PHYSICS].update(&ecs,ECS_PHYSICS,ecs.components[ECS_PHYSICS].map[i],i);
        for(size_t i = 0; i < ecs.components[ECS_HEALTH].size; i++) ecs.components[ECS_HEALTH].update(&ecs,ECS_HEALTH,ecs.components[ECS_HEALTH].map[i],i);
        for(size_t i = 0; i < ecs.components[ECS_TURRET].size; i++) ecs.components[ECS_TURRET].update(&ecs,ECS_TURRET,ecs.components[ECS_TURRET].map[i],i);  

        if(frame%60==0)printf("objective dists: ");
        for(size_t i = 0; i < ecs.components[ECS_VEHICLE].size; i++){
            ecs.components[ECS_VEHICLE].update(&ecs,ECS_VEHICLE,ecs.components[ECS_VEHICLE].map[i],i);
            ecs_position_t* pos = (ecs_position_t*)ecs_component_get_sparse(&ecs,ECS_POSITION,ecs.components[ECS_VEHICLE].map[i]);
            Vector2 a = (Vector2){pos->pos[3],pos->pos[11]};
            float md = FLT_MAX;
            for(size_t j = 0; j < objectives.size();){
                Vector2 b = objectives[i];
                float d = Vector2Distance(a,b);
                if(d < md) md = d;
                if(d < 8){
                    printf("tank %zu collected objective %zu\n",i,j);
                    objectives.erase(objectives.begin()+j);
                }else j++;
            }
            if(frame%60==0)printf("[%zu:%.1f] ",i,md);
        }
        if(frame%60==0)printf("\n");

        if(objectives.size() <= 1){
            for(size_t i = 0; i < 5; i++){
                Vector2 candidate = (Vector2){rand()%800-400.f,rand()%800-400.f};
                float h = heightfield.v[(int)(candidate.y+heightfield.w/2)*heightfield.w+(int)(candidate.x+heightfield.w/2)];
                if(h < 10.) objectives.push_back(candidate);
            }
        }

        tanksim::sensor msg;
        msg.data.clear();
        for(size_t i = 0; i < objectives.size(); i++){
            msg.data.push_back(objectives[i].x);
            msg.data.push_back(objectives[i].y);
        }
        mission.publish(msg);

        BeginMode3D(camera);
        for(size_t i = 0; i < ecs.components[ECS_SENSOR].size; i++) ecs.components[ECS_SENSOR].update(&ecs,ECS_SENSOR,ecs.components[ECS_SENSOR].map[i],i);
        DrawMesh(res.meshes[RES_MESH_TERRAIN],res.materials[RES_MAT_DEFAULT],MatrixIdentity());
        //DrawCube((Vector3){0,0,0},2,2,2,RED);
        for(size_t i = 0; i < RES_MESH_LEN; i++){
            if(res.batch_size[i] > 0){
                DrawMeshInstanced(res.meshes[i], res.materials[RES_MAT_TANK_GREEN], res.transforms[i], res.batch_size[i]);
                //DrawMeshInstanced(res.meshes[RES_MESH_TANK_TURRET], res.materials[RES_MAT_TANK_GREEN], res.transforms[i], res.batch_size[i]);
                //printf("BATCH SIZE %zu\n",res.batch_size[i]);
            } 
            /*for (size_t j = 0; j < res.batch_size[i]; j++) {
                DrawMesh(res.meshes[i], res.materials[RES_MAT_TANK_GREEN], res.transforms[i][j]);
                float16 f = MatrixToFloatV(res.transforms[i][j]);
                printf("transform:\n");
                for(size_t k = 0; k < 16; k++){
                    printf("%f ",f.v[k]);
                    if((k+1)%4==0) printf("\n");
                }
            }*/
            res.batch_size[i] = 0;
        }
        for(size_t i = 0; i < pointcloud.size(); i++){
            DrawPoint3D(pointcloud[i],BLUE);
        }
        //DrawMesh(res.meshes[RES_MESH_TANK],res.materials[RES_MAT_TANK_GREEN],MatrixIdentity());//MatrixMultiply(MatrixIdentity(),MatrixScale(14,14,14)));

        /*Matrix transforms[10];
        for (int i = 0; i < 10; i++) {
            transforms[i] = MatrixTranslate((float)i*2.0f, 0.0f, 0.0f);
        }
        DrawMeshInstanced(GenMeshCube(1,1,1), LoadMaterialDefault(), transforms, 10);*/
        EndMode3D();

        // End drawing
        EndDrawing();
    }

    // De-initialize and close window
    CloseWindow();

    return 0;
}
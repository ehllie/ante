%{
#ifndef AN_PARSER
#define AN_PARSER

#include <stdlib.h>
#include <stdio.h>
#include <tokens.h>
#include <ptree.h>

#ifndef YYSTYPE
#define YYSTYPE Node*
#endif

/* This has no effect when generating a c++ parser */
/* Setting verbose for a c++ parser requires %error-verbose, set in the next section */
#define YYERROR_VERBOSE

#include "yyparser.h"

/* Defined in lexer.cpp */
extern int yylex(...);

namespace ante{
    extern void error(string& msg, const char *fileName, unsigned int row, unsigned int col);
}

void yyerror(const char *msg);

%}

%locations
%error-verbose

%token Ident UserType

/* types */
%token I8 I16 I32 I64 
%token U8 U16 U32 U64
%token Isz Usz F16 F32 F64
%token C8 C32 Bool Void

/* operators */
%token Eq NotEq AddEq SubEq MulEq DivEq GrtrEq LesrEq
%token Or And

/* literals */
%token True False
%token IntLit FltLit StrLit

/* keywords */
%token Return
%token If Elif Else
%token For While Do In
%token Continue Break
%token Import Let Match
%token Data Enum

/* modifiers */
%token Pub Pri Pro Raw
%token Const Ext Noinit Pathogen

/* other */
%token Where Infect Cleanse Ct

/* whitespace */
%token Newline Indent Unindent


/*
    Now to manually fix all shift/reduce conflicts
*/

/*
    Fake precedence rule to allow for a lower precedence
    than Ident in decl context
*/
%nonassoc LOW

%left Ident

%left IntLit FltLit StrLit True False

%left ','

%left Or
%left And     
%left Eq  NotEq GrtrEq LesrEq '<' '>'

%left '+' '-'
%left '*' '/' '%'

%right '^'
%left '.'

%nonassoc '(' '[' Indent
%nonassoc HIGH

/*
    All shift/reduce conflicts should be manually dealt with.
*/
%expect 0
%start top_level_stmt_list
%%

top_level_stmt_list: maybe_newline stmt_list maybe_newline
                   ;

stmt_list: stmt_list nl_stmt Newline {$$ = setNext($1, $2);}
         | stmt_list no_nl_stmt      {$$ = setNext($1, $2);}
         | nl_stmt Newline           {$$ = setRoot($1);}
         | no_nl_stmt                {$$ = setRoot($1);}
         ;

maybe_newline: Newline  %prec Newline
             | %empty   %prec LOW
             ;

/*
 * Statements that will never end with a newline token.
 * Usually statements that require blocks, such as function declarations.
 */
no_nl_stmt: fn_decl
          | data_decl
          | enum_decl
          | while_loop
          | do_while_loop
          | for_loop
          | if_stmt
          ;

/* Statements that can possibly end in an newline */
nl_stmt: var_decl
       | var_assign
       | fn_call
       | ret_stmt
       | let_binding
       ;

ident: Ident {$$ = (Node*)lextxt;}
     ;

usertype: UserType {$$ = (Node*)lextxt;}
        ;

intlit: IntLit {$$ = mkIntLitNode(lextxt);}
      ;

fltlit: FltLit {$$ = mkFltLitNode(lextxt);}
      ;

strlit: StrLit {$$ = mkStrLitNode(lextxt);}
      ;

lit_type: I8       {$$ = mkTypeNode(Tok_I8,  (char*)"");}
        | I16      {$$ = mkTypeNode(Tok_I16, (char*)"");}
        | I32      {$$ = mkTypeNode(Tok_I32, (char*)"");}
        | I64      {$$ = mkTypeNode(Tok_I64, (char*)"");}
        | U8       {$$ = mkTypeNode(Tok_U8,  (char*)"");}
        | U16      {$$ = mkTypeNode(Tok_U16, (char*)"");}
        | U32      {$$ = mkTypeNode(Tok_U32, (char*)"");}
        | U64      {$$ = mkTypeNode(Tok_U64, (char*)"");}
        | Isz      {$$ = mkTypeNode(Tok_Isz, (char*)"");}
        | Usz      {$$ = mkTypeNode(Tok_Usz, (char*)"");}
        | F16      {$$ = mkTypeNode(Tok_F16, (char*)"");}
        | F32      {$$ = mkTypeNode(Tok_F32, (char*)"");}
        | F64      {$$ = mkTypeNode(Tok_F64, (char*)"");}
        | C8       {$$ = mkTypeNode(Tok_C8,  (char*)"");}
        | C32      {$$ = mkTypeNode(Tok_C32, (char*)"");}
        | Bool     {$$ = mkTypeNode(Tok_Bool, (char*)"");}
        | Void     {$$ = mkTypeNode(Tok_Void, (char*)"");}
        | usertype %prec UserType {$$ = mkTypeNode(Tok_UserType, (char*)$1);}
        | ident    %prec Ident {$$ = mkTypeNode(Tok_Ident, (char*)$1);}
        ;

type: type '*'                {$$ = mkTypeNode('*', (char*)"", $1);}
    | type '[' maybe_expr ']' {$$ = mkTypeNode('[', (char*)"", $1);}
    | type '(' type_expr ')'  {$$ = mkTypeNode('(', (char*)"", $1);}  /* f-ptr w/ params*/
    | type '(' ')'            {$$ = mkTypeNode('(', (char*)"", $1);}  /* f-ptr w/out params*/
    | '(' type_expr ')'       {$$ = $1;}
    | lit_type                {$$ = $1;}
    ;

type_expr_: type_expr_ ',' type {$$ = setNext($1, $3);}
          | type_expr_ '|' type
          | type                {$$ = setRoot($1);}
          ;

type_expr: type_expr_ {$$ = getRoot();}


modifier: Pub      {$$ = mkModNode(Tok_Pub);} 
        | Pri      {$$ = mkModNode(Tok_Pri);}
        | Pro      {$$ = mkModNode(Tok_Pro);}
        | Raw      {$$ = mkModNode(Tok_Raw);}
        | Const    {$$ = mkModNode(Tok_Const);}
        | Ext      {$$ = mkModNode(Tok_Ext);}
        | Noinit   {$$ = mkModNode(Tok_Noinit);}
        | Pathogen {$$ = mkModNode(Tok_Pathogen);}
        ;

modifier_list_: modifier_list_ modifier {$$ = setNext($1, $2);}
              | modifier {$$ = setRoot($1);}
              ;

modifier_list: modifier_list_ {$$ = getRoot();}
             ;


var_decl: modifier_list type_expr ident '=' expr  %prec Ident {$$ = mkVarDeclNode((char*)$3, $1, $2, $5);}
        | modifier_list type_expr ident           %prec LOW   {$$ = mkVarDeclNode((char*)$3, $1, $2,  0);}
        | type_expr ident '=' expr                %prec Ident {$$ = mkVarDeclNode((char*)$2, 0,  $1, $4);}
        | type_expr ident                         %prec LOW   {$$ = mkVarDeclNode((char*)$2, 0,  $1,  0);}
        ;

let_binding: Let modifier_list type_expr ident '=' expr  {$$ = mkLetBindingNode((char*)$3, $2, $3, $6);}
           | Let modifier_list ident '=' expr            {$$ = mkLetBindingNode((char*)$2, $2, 0,  $5);}
           | Let type_expr ident '=' expr                {$$ = mkLetBindingNode((char*)$3, 0,  $2, $5);}
           | Let ident '=' expr                          {$$ = mkLetBindingNode((char*)$2, 0,  0,  $4);}
           ;

/* TODO: change arg1 to require node* instead of char* */
var_assign: ref_val '=' expr {$$ = mkVarAssignNode($1, $3);}
          | ref_val AddEq expr {$$ = mkVarAssignNode($1, mkBinOpNode('+', mkUnOpNode('*', $1), $3));}
          | ref_val SubEq expr {$$ = mkVarAssignNode($1, mkBinOpNode('-', mkUnOpNode('*', $1), $3));}
          | ref_val MulEq expr {$$ = mkVarAssignNode($1, mkBinOpNode('*', mkUnOpNode('*', $1), $3));}
          | ref_val DivEq expr {$$ = mkVarAssignNode($1, mkBinOpNode('/', mkUnOpNode('*', $1), $3));}
          ;

usertype_list: usertype_list ',' usertype {$$ = setNext($1, $3);}
             | usertype {$$ = setRoot($1);}
             ;

generic: '<' usertype_list '>' {$$ = getRoot();}
       ;

data_decl: modifier_list Data usertype type_decl_block         {$$ = mkDataDeclNode((char*)$3, $4);}
         | modifier_list Data usertype generic type_decl_block {$$ = mkDataDeclNode((char*)$3, $5);}
         | Data usertype type_decl_block                       {$$ = mkDataDeclNode((char*)$2, $3);}
         | Data usertype generic type_decl_block               {$$ = mkDataDeclNode((char*)$2, $4);}
         ;

type_decl: type_expr ident {$$ = mkNamedValNode(mkVarNode((char*)$2), $1);}
         | type_expr       {$$ = mkNamedValNode(0, $1);}
         | enum_decl /* TODO */
         ;

type_decl_list: type_decl_list Newline type_decl  {$$ = setNext($1, $3);}
              | type_decl                         {$$ = setRoot($1);}
              ;

type_decl_block: Indent type_decl_list Unindent  {$$ = getRoot();}
               ;

/* Specifying an enum member's value */
val_init_list: val_init_list Newline usertype
             | val_init_list Newline usertype '=' expr
             | usertype '=' expr
             | usertype
             ;

enum_block: Indent val_init_list Unindent
          ;

enum_decl: modifier_list Enum usertype enum_block  {$$ = NULL;}
         | Enum usertype enum_block                {$$ = NULL;}
         | modifier_list Enum enum_block           {$$ = NULL;}
         | Enum enum_block                         {$$ = NULL;}
         ;

block: Indent stmt_list no_nl_stmt Unindent {setNext($2, $3); $$ = getRoot();}
     | Indent stmt_list nl_stmt Unindent    {setNext($2, $3); $$ = getRoot();}
     | Indent no_nl_stmt Unindent           {$$ = $2;}
     | Indent nl_stmt Unindent              {$$ = $2;}
     ;

ident_list: ident_list ident  {$$ = setNext($1, mkVarNode((char*)$2));}
          | ident             {$$ = setRoot(mkVarNode((char*)$1));}
          ;

/* 
 * In case of multiple parameters declared with a single type, eg i32 a b c
 * The next parameter should be set to the first in the list, (the one returned by getRoot()),
 * but the variable returned must be the last in the last, in this case $4
 */
params: params ',' type_expr ident_list {$$ = setNext($1, mkNamedValNode(getRoot(), $3));}
      | type_expr ident_list            {$$ = setRoot(mkNamedValNode(getRoot(), $1));}
      ;

maybe_params: params {$$ = getRoot();}
            | %empty {$$ = NULL;}
            ;

fn_decl: modifier_list type_expr ident ':' maybe_params block                    {$$ = mkFuncDeclNode((char*)$3, $1, $2, $5, $6);}
       | modifier_list type_expr ident '(' maybe_expr ')' ':' maybe_params block {$$ = mkFuncDeclNode((char*)$3, $1, $2, $8, $9);}
       | type_expr ident ':' maybe_params block                                  {$$ = mkFuncDeclNode((char*)$2, 0,  $1, $4, $5);}
       | type_expr ident '(' maybe_expr ')' ':' maybe_params block               {$$ = mkFuncDeclNode((char*)$2, 0,  $1, $7, $8);}
       ;

fn_call: ident tuple {$$ = mkFuncCallNode((char*)$1, $2);}
       ;

ret_stmt: Return expr {$$ = mkRetNode($2);}
        ;

elif_list: elif_list Elif expr block {$$ = setElse((IfNode*)$1, (IfNode*)mkIfNode($3, $4));}
         | Elif expr block {$$ = setRoot(mkIfNode($2, $3));}
         ;

maybe_elif_list: elif_list Else block {$$ = setElse((IfNode*)$1, (IfNode*)mkIfNode(NULL, $3));}
               | elif_list {$$ = $1;}
               | Else block {$$ = setRoot(mkIfNode(NULL, $2));}
               | %empty {$$ = setRoot(NULL);}
               ;

if_stmt: If expr block maybe_elif_list {$$ = mkIfNode($2, $3, (IfNode*)getRoot());}
       ;

while_loop: While expr block {$$ = NULL;}
          ;

do_while_loop: Do While expr block {$$ = NULL;}
             ;

for_loop: For var_decl In expr block {$$ = NULL;}
        ;

var: ident '[' expr ']'              {$$ = 0;}/*TODO: arrays*/
   | ident               %prec Ident {$$ = mkVarNode((char*)$1);}
   ;

ref_val: '&' ref_val         {$$ = mkUnOpNode('&', $2);}
       | '*' ref_val         {$$ = mkUnOpNode('*', $2);}
       | ident '[' expr ']'  
       | ident  %prec Ident  {$$ = mkRefVarNode((char*)$1);}
       ;

val: fn_call                 {$$ = $1;}
   | tuple                   {$$ = $1;}
   | array                   {$$ = $1;}
   | Indent nl_expr Unindent {$$ = $2;}
   | unary_op                {$$ = $1;}
   | var                     {$$ = $1;}
   | intlit                  {$$ = $1;}
   | fltlit                  {$$ = $1;}
   | strlit                  {$$ = $1;}
   | True                    {$$ = mkBoolLitNode(1);}
   | False                   {$$ = mkBoolLitNode(0);}
   ;

tuple: '(' expr_list ')'  {$$ = mkTupleNode($2);}
     | '(' ')'            {$$ = mkTupleNode(0);}
     ;

array: '[' expr_list ']' {$$ = mkArrayNode($2);}
     | '[' ']'           {$$ = mkArrayNode(0);}
     ;

maybe_expr: expr    {$$ = $1;}
          | %empty  {$$ = NULL;}
          ;

expr_list: expr_list_p {$$ = getRoot();}
         ;

expr_list_p: expr_list_p ',' expr    {$$ = setNext($1, $3);}
           | expr                    {$$ = setRoot($1);}
           ;

unary_op: '*' val  {$$ = mkUnOpNode('*', $2);}
        | '&' val  {$$ = mkUnOpNode('&', $2);}
        | '-' val  {$$ = mkUnOpNode('-', $2);}
        ;

expr: expr '+' expr     {$$ = mkBinOpNode('+', $1, $3);}
    | expr '-' expr     {$$ = mkBinOpNode('-', $1, $3);}
    | expr '*' expr     {$$ = mkBinOpNode('*', $1, $3);}
    | expr '/' expr     {$$ = mkBinOpNode('/', $1, $3);}
    | expr '%' expr     {$$ = mkBinOpNode('%', $1, $3);}
    | expr '<' expr     {$$ = mkBinOpNode('<', $1, $3);}
    | expr '>' expr     {$$ = mkBinOpNode('>', $1, $3);}
    | expr '^' expr     {$$ = mkBinOpNode('^', $1, $3);}
    | expr '.' expr     {$$ = mkBinOpNode('.', $1, $3);}
    | expr Eq expr      {$$ = mkBinOpNode(Tok_Eq, $1, $3);}
    | expr NotEq expr   {$$ = mkBinOpNode(Tok_NotEq, $1, $3);}
    | expr GrtrEq expr  {$$ = mkBinOpNode(Tok_GrtrEq, $1, $3);}
    | expr LesrEq expr  {$$ = mkBinOpNode(Tok_LesrEq, $1, $3);}
    | expr Or expr      {$$ = mkBinOpNode(Tok_Or, $1, $3);}
    | expr And expr     {$$ = mkBinOpNode(Tok_And, $1, $3);}
    | val               {$$ = $1;}
    ;


/* nl_expr is used in expression blocks and can span multiple lines */
nl_expr: nl_expr_list {$$ = getRoot();}
       ;

nl_expr_list: nl_expr_list ',' maybe_newline expr_block_p {$$ = setNext($1, $4);}
            | expr_block_p                                {$$ = setRoot($1);}
            ;

expr_block_p: expr_block_p '+' maybe_newline expr_block_p     {$$ = mkBinOpNode('+', $1, $4);}
         | expr_block_p '-' maybe_newline expr_block_p     {$$ = mkBinOpNode('-', $1, $4);}
         | expr_block_p '*' maybe_newline expr_block_p     {$$ = mkBinOpNode('*', $1, $4);}
         | expr_block_p '/' maybe_newline expr_block_p     {$$ = mkBinOpNode('/', $1, $4);}
         | expr_block_p '%' maybe_newline expr_block_p     {$$ = mkBinOpNode('%', $1, $4);}
         | expr_block_p '<' maybe_newline expr_block_p     {$$ = mkBinOpNode('<', $1, $4);}
         | expr_block_p '>' maybe_newline expr_block_p     {$$ = mkBinOpNode('>', $1, $4);}
         | expr_block_p '^' maybe_newline expr_block_p     {$$ = mkBinOpNode('^', $1, $4);}
         | expr_block_p '.' maybe_newline expr_block_p     {$$ = mkBinOpNode('.', $1, $4);}
         | expr_block_p Eq maybe_newline  expr_block_p     {$$ = mkBinOpNode(Tok_Eq, $1, $4);}
         | expr_block_p NotEq maybe_newline expr_block_p   {$$ = mkBinOpNode(Tok_NotEq, $1, $4);}
         | expr_block_p GrtrEq maybe_newline expr_block_p  {$$ = mkBinOpNode(Tok_GrtrEq, $1, $4);}
         | expr_block_p LesrEq maybe_newline expr_block_p  {$$ = mkBinOpNode(Tok_LesrEq, $1, $4);}
         | expr_block_p Or maybe_newline expr_block_p      {$$ = mkBinOpNode(Tok_Or, $1, $4);}
         | expr_block_p And maybe_newline expr_block_p     {$$ = mkBinOpNode(Tok_And, $1, $4);}
         | val                                             {$$ = $1;}
         ;
%%

void yy::parser::error(const location& loc, const string& msg){
    ante::error(msg.c_str(), yylexer->fileName, yylexer->getRow(), yylexer->getCol());
}

#endif
